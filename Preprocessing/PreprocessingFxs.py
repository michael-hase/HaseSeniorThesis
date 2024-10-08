import os
import subprocess
import nibabel as nib
from pathlib import Path
import numpy as np
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from scipy.signal import medfilt


# Load nifti file and return image data and affine

def load_nii(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine

# Perform N4ITK bias field correction on image
def bias_field_correction(src_path: str | Path, dst_path: str | Path) -> None:
    print("N4ITK on: ", src_path)
    n4 = N4BiasFieldCorrection()
    n4.inputs.input_image = src_path
    n4.inputs.output_image = dst_path
    n4.inputs.dimension = 3
    n4.inputs.n_iterations = [100, 100, 60, 40]
    n4.inputs.shrink_factor = 3
    n4.inputs.convergence_threshold = 1e-4
    n4.inputs.bspline_fitting_distance = 300
    n4.run()

# Save data as nifti file

def save_nii(data: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    nib.save(nib.Nifti1Image(data, affine), path)

# Denoise using w/ median filter
def denoise(volume: np.ndarray, kernel_size: int = 3):
    return medfilt(volume, kernel_size)

# Rescale the intensity of the image volume

def rescale_intensity(volume: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5),
                      bins_num: int = 256) -> np.ndarray:
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentiles[0])
    max_value = np.percentile(obj_volume, percentiles[1])

    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume

    return volume

# Perform histogram equalization of image volume

def equalize_hist(volume: np.ndarray, bins_num=256) -> np.ndarray:
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num, density=True)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


def run_command(cmd):
    print(f"Running command: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {cmd}")
        print(f"Command failed with exit code {e.returncode}")

# Run preprocessing pipeline on MRI scans

def process_subject(subject_dir, overwrite=False):
    # Create processed folder inside the subject directory
    processed_dir = os.path.join(subject_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)  # Check to make sure directory exisits

    failed_records_log = open(Path(processed_dir, "failed_records.log"), "w+")
    for root, dirs, files in os.walk(subject_dir):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nii_file = os.path.join(root, file)
                filename = f"{str(file).split('.')[0]}.nii.gz"

                # Paths for processing step outputs
                bet_output = os.path.join(processed_dir, f"BET_{filename}")
                n4_output = os.path.join(processed_dir, f"N4_{filename}")
                alignment_output = os.path.join(processed_dir, f"alignment_{filename}")
                denoised_output = os.path.join(processed_dir, f"denoised_{filename}")
                rescaled_output = os.path.join(processed_dir, f"rescaled_{filename}")
                equalized_output = os.path.join(processed_dir, f"equalized_{filename}")

                try:
                    # Step 1: Brain Extraction using FSL's BET
                    if not overwrite and not os.path.exists(bet_output):
                        bet_command = f"bet {nii_file} {bet_output} -f 0.5 -g 0"
                        run_command(bet_command)

                    # Step 2: N4 Bias Field Correction using ANTs
                    if not overwrite and not os.path.exists(n4_output):
                        n4_command = f"N4BiasFieldCorrection -i {bet_output} -o {n4_output}"
                        run_command(n4_command)

                    # Step 3: Alignment to MNI space using FSL's FLIRT
                    if not overwrite and not os.path.exists(alignment_output):
                        flirt_command = f"flirt -in {n4_output} -ref $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz -out {alignment_output} -dof 6"
                        run_command(flirt_command)

                    # Step 4: Denoising
                    if not overwrite and not os.path.exists(denoised_output):
                        volume, affine = load_nii(alignment_output)
                        volume = denoise(volume)
                        save_nii(volume, affine, denoised_output)

                    # Step 5: Rescaling intensity
                    if not overwrite and not os.path.exists(rescaled_output):
                        volume = rescale_intensity(volume)
                        save_nii(volume, affine, rescaled_output)

                    # Step 6: Histogram equalization
                    if not overwrite and not os.path.exists(equalized_output):
                        volume = equalize_hist(volume)
                        save_nii(volume, affine, equalized_output)

                except Exception as e:
                    print(f"Failed to process {nii_file}")
                    failed_records_log.write(f"{nii_file},{e}\n")
                    failed_records_log.flush()
                    continue

    failed_records_log.close()
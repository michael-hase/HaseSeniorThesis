# Load Packages
import os
import subprocess
import nibabel as nib
from pathlib import Path
import numpy as np
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from scipy.signal import medfilt

# Add ANTs installation to the PATH
os.environ['PATH'] = '/Users/michaelhase/ants-2.5.3/bin:' + os.environ['PATH']

# Define base directory for input/output
base_dir = "/Users/michaelhase/Desktop/SeniorThesis/data/ADNI"

# List subjects
subject_dirs = [
    "002_S_0816", "018_S_0450", "036_S_1135", "094_S_1027", "130_S_0423",
    "002_S_0954", "018_S_0633", "036_S_1240", "094_S_1090", "130_S_0505",
    "003_S_0907", "018_S_0682", "037_S_0467", "094_S_1293", "130_S_0783",
    "003_S_0908", "020_S_0097", "037_S_0501", "094_S_1397", "130_S_0886",
    "003_S_0981", "020_S_0213", "037_S_0552", "094_S_1398", "130_S_0956",
    "003_S_1057", "021_S_0231", "037_S_0588", "094_S_1402", "130_S_0969",
    "005_S_0222", "021_S_0332", "037_S_0627", "094_S_1417", "130_S_1200",
    "005_S_0448", "021_S_0642", "037_S_1225", "099_S_0054", "130_S_1201",
    "006_S_0498", "021_S_0647", "037_S_1421", "099_S_0060", "130_S_1290",
    "006_S_0675", "022_S_0004", "041_S_0282", "099_S_0111", "130_S_1337",
    "006_S_0681", "022_S_0544", "041_S_0446", "099_S_0880", "132_S_0987",
    "006_S_0731", "022_S_0750", "041_S_0549", "100_S_0006", "133_S_0525",
    "007_S_0070", "023_S_0078", "041_S_0598", "100_S_0015", "133_S_0629",
    "007_S_0414", "023_S_0855", "041_S_1412", "100_S_0035", "133_S_0638",
    "009_S_0751", "023_S_1126", "041_S_1423", "100_S_0047", "133_S_0727",
    "009_S_0842", "023_S_1247", "051_S_1040", "100_S_0069", "133_S_0771",
    "009_S_0862", "024_S_1393", "051_S_1072", "100_S_0190", "133_S_0792",
    "009_S_1030", "027_S_0417", "051_S_1331", "100_S_0296", "133_S_0912",
    "010_S_0067", "027_S_0461", "052_S_1054", "100_S_0995", "133_S_0913",
    "010_S_0472", "027_S_0485", "052_S_1168", "109_S_0950", "136_S_0194",
    "010_S_0904", "027_S_1213", "053_S_1044", "109_S_0967", "136_S_0195",
    "011_S_0022", "027_S_1277", "057_S_0941", "109_S_1014", "136_S_0579",
    "011_S_0168", "027_S_1387", "057_S_1265", "109_S_1114", "136_S_0695",
    "011_S_0856", "029_S_0836", "062_S_0578", "109_S_1183", "136_S_0873",
    "012_S_0712", "029_S_0999", "062_S_0768", "109_S_1343", "136_S_0874",
    "012_S_0720", "029_S_1215", "062_S_1099", "114_S_0410", "136_S_1227",
    "012_S_0932", "031_S_0294", "067_S_0019", "114_S_0458", "137_S_0669",
    "012_S_1009", "031_S_0321", "067_S_0038", "114_S_1103", "137_S_0825",
    "012_S_1033", "031_S_0351", "067_S_0056", "116_S_1243", "137_S_1426",
    "012_S_1165", "031_S_0568", "067_S_0059", "121_S_1322", "141_S_0696",
    "012_S_1292", "031_S_0830", "067_S_0077", "123_S_0050", "141_S_0717",
    "012_S_1321", "031_S_0867", "067_S_0098", "123_S_0072", "141_S_0726",
    "013_S_0240", "031_S_1066", "067_S_0176", "123_S_0088", "141_S_0767",
    "013_S_0325", "031_S_1209", "067_S_0177", "123_S_0106", "141_S_0790",
    "013_S_0860", "032_S_0095", "067_S_0284", "123_S_0113", "141_S_0810",
    "013_S_1120", "032_S_0187", "067_S_0290", "123_S_0390", "141_S_0851",
    "013_S_1186", "032_S_0479", "067_S_0336", "126_S_0606", "141_S_0852",
    "013_S_1275", "032_S_0677", "067_S_0607", "126_S_0709", "141_S_0853",
    "014_S_1095", "032_S_0718", "068_S_0442", "126_S_1187", "141_S_0915",
    "016_S_0354", "032_S_0978", "068_S_0872", "127_S_0393", "141_S_0982",
    "016_S_1028", "032_S_1101", "072_S_0315", "127_S_0684", "141_S_1004",
    "016_S_1117", "033_S_0511", "073_S_0311", "127_S_1427", "141_S_1052",
    "018_S_0043", "033_S_0725", "073_S_0312", "128_S_0863", "141_S_1094",
    "018_S_0055", "033_S_1086", "073_S_0518", "128_S_0947", "141_S_1137",
    "018_S_0057", "033_S_1116", "073_S_0746", "128_S_1043", "141_S_1152",
    "018_S_0080", "033_S_1279", "082_S_0928", "128_S_1088", "141_S_1245",
    "018_S_0087", "033_S_1284", "082_S_1119", "128_S_1148", "141_S_1255",
    "018_S_0142", "033_S_1309", "082_S_1256", "128_S_1242", "941_S_1197",
    "018_S_0155", "036_S_0576", "082_S_1377", "128_S_1407", "941_S_1311",
    "018_S_0369", "036_S_0748", "094_S_0531", "130_S_0102",
    "018_S_0406", "036_S_0976", "094_S_0692", "130_S_0232"
]

# Define functions for scan preprocessing

def load_nii(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine


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


def save_nii(data: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    nib.save(nib.Nifti1Image(data, affine), path)


def denoise(volume: np.ndarray, kernel_size: int = 3):
    return medfilt(volume, kernel_size)


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


def main():
    for subject in subject_dirs:
        subject_dir = os.path.join(base_dir, subject)
        print(f"Processing subject: {subject}")
        process_subject(subject_dir)


if __name__ == "__main__":
    main()
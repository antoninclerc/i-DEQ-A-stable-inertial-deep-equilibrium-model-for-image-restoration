import tarfile
import os

def extract_first_n(archive_path, output_dir, n):
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(archive_path, "r|xz") as tar:  # streaming mode
        count = 0

        for member in tar:
            if member.isfile() and member.name.endswith(".h5"):

                member.name = os.path.basename(member.name)
                tar.extract(member, output_dir)

                count += 1
                print("extracted:", count)

                if count >= n:
                    break


train_archive = "Fast_MRI_data/knee_singlecoil_train.tar.xz"
val_archive = "Fast_MRI_data/knee_singlecoil_val.tar.xz"

extract_first_n(train_archive, "DATA/singlecoil_train", 500)
extract_first_n(val_archive, "DATA/singlecoil_val", 10)
extract_first_n(val_archive, "DATA/singlecoil_test", 50)
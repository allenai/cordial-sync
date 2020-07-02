"""
Script used to download and extract:

* `ai2thor_builds` - Our custom AI2-THOR build (necessary for running any experiments),
* `data` - Our evaluation data (necessary to reproduce the plots/tables reported in our paper), and
* `trained_models` - Our trained model checkpoints.

When running
 ```bash
python rl_multi_agent/scripts/download_evaluation_data.py ai2thor_builds
```
this script downloads all of the above. If you'd prefer to only download a single one
of the above directories simply specify this on the command line, e.g.
```bash
python rl_multi_agent/scripts/download_evaluation_data.py ai2thor_builds
```
will only download the `ai2thor_builds` directory.
"""

import glob
import hashlib
import os
import platform
import shutil
import sys
from zipfile import ZipFile

from constants import PLATFORM, PROJECT_TOP_DIR

DIR_TO_DOWNLOAD_ID = {
    "ai2thor_builds": "1zRbOYb-K07R7Bb1vMBGWdPGqXYY-adMY",
    "data": "1iz_zV74ZIdZd9UwYDvomKPRgqwpPctEi",
    "trained_models": "1vLckX20fnImvoxugZd812fKi2JVKwC0Z",
}

DIR_TO_MD5 = {
    "ai2thor_builds": "84201a23b87b60771ac16e7814cd28a0",
    "data": "fbbad25bb7d0494fe9a818e496630522",
    "trained_models": "b3d4b0208515a0fc59096e2659bb8b19",
}

DIR_TO_NAME = {
    "ai2thor_builds": "AI2-THOR builds",
    "data": "evaluation data",
    "trained_models": "model checkpoints",
}


def get_md5_hash(file_name: str):
    md5_obj = hashlib.md5()
    with open(file_name, "rb") as f:
        buf = f.read()
        md5_obj.update(buf)
    return md5_obj.hexdigest()


if platform.system() == "Linux":
    DOWNLOAD_TEMPLATE = """wget --quiet --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={}" -O {}"""
elif platform.system() == "Darwin":
    DOWNLOAD_TEMPLATE = """wget --quiet --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={}" -O {}"""
else:
    raise NotImplementedError("{} is not supported".format(PLATFORM))

if __name__ == "__main__":

    if len(sys.argv) == 1 or sys.argv[1] == "all":
        print("Beginning process of downloading, unzipping, and moving all data.\n")
        download_dirs = sorted(list(DIR_TO_DOWNLOAD_ID.keys()))
    else:
        download_dirs = [sys.argv[1].lower()]
        assert (
            download_dirs[0] in DIR_TO_DOWNLOAD_ID
        ), "Do not know how to download {} directory.".format(download_dirs[0])

    for rel_download_dir in download_dirs:
        print("Now populating {} directory.".format(rel_download_dir))

        download_id = DIR_TO_DOWNLOAD_ID[rel_download_dir]
        abs_download_dir = os.path.join(PROJECT_TOP_DIR, rel_download_dir)
        zip_expected_md5 = DIR_TO_MD5[rel_download_dir]

        os.makedirs(abs_download_dir, exist_ok=True)

        tmp_directory = os.path.join("/tmp", download_id)

        assert os.path.exists("/tmp")
        os.makedirs(tmp_directory, exist_ok=True)

        tmp_zip_path = os.path.join(tmp_directory, "to_unzip.zip")
        tmp_unzip_path = os.path.join(tmp_directory, "unzipped_contents")

        if (
            os.path.exists(tmp_zip_path)
            and get_md5_hash(tmp_zip_path) == DIR_TO_MD5[rel_download_dir]
        ):
            print(
                "{} already exists and has correct hash, skipping download.".format(
                    tmp_zip_path
                )
            )

        else:
            print(
                "Downloading archive to temporary directory {}...".format(tmp_zip_path)
            )

            os.system(DOWNLOAD_TEMPLATE.format(download_id, download_id, tmp_zip_path))

            print("Downloaded.")

            if (not os.path.exists(tmp_zip_path)) or get_md5_hash(
                tmp_zip_path
            ) != DIR_TO_MD5[rel_download_dir]:
                print(
                    "Could not download contents of {}, this is likely an error. Skipping...".format(
                        rel_download_dir
                    )
                )
                continue

        print("Unzipping to temporary file {}...".format(tmp_unzip_path))

        with ZipFile(tmp_zip_path, "r") as zip_file:
            zip_file.extractall(tmp_unzip_path)

        print("Unzipped.".format(tmp_unzip_path))

        if os.path.exists(os.path.join(tmp_unzip_path, rel_download_dir)):
            tmp_unzip_path = os.path.join(tmp_unzip_path, rel_download_dir)

        print("Moving unzipped contents to {}".format(abs_download_dir))

        for from_path in glob.glob(os.path.join(tmp_unzip_path, "*")):
            if "_MACOS" in from_path:
                continue
            to_path = os.path.join(abs_download_dir, os.path.basename(from_path))

            if os.path.exists(to_path):
                while True:
                    should_proceed_str = (
                        input(
                            "\nMoving {} to {} but this will cause {} to be overwritten. "
                            "Overwrite? (y/n)".format(from_path, to_path, to_path)
                        )
                        .strip()
                        .lower()
                    )

                    if should_proceed_str in ["yes", "y"]:
                        should_proceed = True
                        break
                    elif should_proceed_str in ["no", "n"]:
                        should_proceed = False
                        break
                    else:
                        print(
                            "Unrecognized input '{}', please try again.".format(
                                should_proceed_str
                            )
                        )
            else:
                should_proceed = True

            if not should_proceed:
                print("Not overwritting {} and continuing.".format(abs_download_dir))
                continue

            print("Moving extracted path {} to {}".format(from_path, to_path))
            if os.path.exists(to_path):
                if os.path.isdir(to_path):
                    shutil.rmtree(to_path)
                else:
                    os.remove(to_path)
            shutil.move(from_path, to_path)

        print("Moving complete!".format(rel_download_dir))
        print()

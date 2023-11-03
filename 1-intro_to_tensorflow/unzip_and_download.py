from __future__ import annotations
import os
import urllib.request
import zipfile

FILE_URLS = [
    "https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip",
    "https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip"
]


def download_and_unzip(file_urls: list[os.PathLike | str]) -> None:
    """
    logic:
    -- Check if the .zip file already exists in the current directory
        -- no .zip file -> unzipped? continue : unzip it and delete .zip
        -- .zip file -> unzipped? continue : unzip it and delete .zip
    --
    """
    for file_url in file_urls:
        file_name = os.path.basename(file_url)  # horse-or-human.zip, validation-horse-or-human.zip
        dir_name = file_name.strip(".zip")

        if os.path.exists(file_name):  # .zip file exists in cwd
            if os.path.exists(dir_name):  # path exists and has been unzipped
                print(f"/{dir_name} already exists - {file_name} has been removed.")
                os.remove(file_name)  # clean up
            else:  # .zip exists in cwd but not unzipped
                print(f"Downloading {file_name}...")
                try:
                    with zipfile.ZipFile(file_name, 'r') as zip_ref:
                        zip_ref.extractall(dir_name)
                        print(f"{file_name} has been successfully extracted.\n Deleting {file_name}...")
                        os.remove(file_name)  # Remove the ZIP file
                        print(f"{file_name} has been deleted.")
                except Exception as e:
                    print(f"Failed to extract or remove {file_name}: {e}")

        elif os.path.exists(dir_name):  # no .zip, but dir exists:
            print(f"file already downloaded -> /{dir_name}")
            continue
        else:  # no .zip, no dir
            print(f"Downloading {file_name}...")
            try:
                urllib.request.urlretrieve(file_url, file_name)  # try to download the file
                print(f"{file_name} downloaded successfully.")
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")

            try:
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(dir_name)
                    print(f"{file_name} has been successfully extracted.")
                os.remove(file_name)  # Remove the ZIP file
                print(f"{file_name} has been removed.")
            except Exception as e:
                print(f"Failed to extract or remove {file_name}: {e}")

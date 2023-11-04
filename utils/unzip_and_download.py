from __future__ import annotations
import os
import urllib.request
import zipfile


def download_and_unzip(file_urls: list[os.PathLike | str] | os.PathLike, dest_dir: os.PathLike | str = None) -> None:
    """
    logic:
    -- Check if the .zip file already exists in the current directory
        -- no .zip file -> unzipped? continue : unzip it and delete .zip
        -- .zip file -> unzipped? continue : unzip it and delete .zip
    --
    """
    if not isinstance(file_urls, list):
        file_urls = [file_urls]
    for file_url in file_urls:
        file_name = os.path.basename(file_url)  # horse-or-human.zip, validation-horse-or-human.zip
        if dest_dir is None:
            dest_dir = file_name.strip(".zip")

        if os.path.exists(file_name):  # .zip file exists in cwd
            if os.path.exists(dest_dir):  # path exists and has been unzipped
                print(f"/{dest_dir} already exists - {file_name} has been removed.")
                os.remove(file_name)  # clean up
            else:  # .zip exists in cwd but not unzipped
                print(f"Downloading {file_name}...")
                try:
                    with zipfile.ZipFile(file_name, 'r') as zip_ref:
                        zip_ref.extractall(dest_dir)
                        print(f"{file_name} has been successfully extracted.\n Deleting {file_name}...")
                        os.remove(file_name)  # Remove the ZIP file
                        print(f"{file_name} has been deleted.")
                except Exception as e:
                    print(f"Failed to extract or remove {file_name}: {e}")

        elif os.path.exists(dest_dir):  # no .zip, but dir exists:
            print(f"file already downloaded -> /{dest_dir}")
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
                    zip_ref.extractall(dest_dir)
                    print(f"{file_name} has been successfully extracted.")
                os.remove(file_name)  # Remove the ZIP file
                print(f"{file_name} has been removed.")
            except Exception as e:
                print(f"Failed to extract or remove {file_name}: {e}")

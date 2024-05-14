"""
Uploads a file to a specific folder in Google Drive preserving the original directory structure.

usage: drive_upload.py <Google Drive folder ID> <local file path>
example usage: drive_upload.py 0B5XXXXY9KddXXXXXXXA2c3ZXXXX /path/to/my/file
"""
import sys
import glob
import os
import time
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pydrive2.settings import LoadSettingsFile


def upload_file_path(
        drive: GoogleDrive,
        file_path: str,
        file_name: str,
        parent_folder_id: str):
    curr_parent = parent_folder_id
    dir_tree = file_path.split(os.path.sep)
    for idx, name in enumerate(dir_tree):
        file_list = drive.ListFile(
            {
                "q": '"{}" in parents and title="{}" and trashed=false'.format(
                    curr_parent, name
                )
            }
        ).GetList()

        if len(file_list) > 1:
            print(
                f"More than one match for name {os.path.sep.join(dir_tree[:idx+1], name)}!")
            exit(1)
        elif len(file_list) == 1:  # if file/folder already exists
            file = file_list[0]
            if name != file_name:  # this is a folder
                curr_parent = file["id"]
            else:
                file.SetContentFile(file_path)
        else:
            metadata = {"parents": [
                {"kind": "drive#fileLink", "id": curr_parent}]}
            metadata["title"] = name
            if name != file_name:  # this is a folder
                metadata["mimeType"] = "application/vnd.google-apps.folder"
            file = drive.CreateFile(metadata)
            if name == file_name:
                file.SetContentFile(file_path)

        file.Upload()
        if name != file_name:  # this is a folder, so we update the parent id for the next iter
            curr_parent = file["id"]
        # avoiding rate limit
        time.sleep(0.1)


def main():
    gauth = GoogleAuth()
    gauth.settings = LoadSettingsFile(filename="settings.yaml")
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    if len(sys.argv) < 2:
        print("usage: drive_upload.py <Google Drive folder ID> <local file path>")
        exit(1)

    parent_folder_id = sys.argv[1]
    glob_patterns = sys.argv[2:]
    print(f"{len(glob_patterns)} glob patterns found")

    count = 0
    for pattern in glob_patterns:
        for file_path in glob.glob(pattern):
            file_name = os.path.basename(file_path)
            upload_file_path(drive, file_path, file_name, parent_folder_id)
            count += 1
    print(f"Uploaded {count} matched files from glob patterns")


if __name__ == "__main__":
    main()

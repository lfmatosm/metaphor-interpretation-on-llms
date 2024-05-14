#!/bin/bash

DRIVE_FOLDER_ID="<put_your_google_drive_folder_id_here>"
python src/utils/drive_upload.py "$DRIVE_FOLDER_ID" src/**/*.py src/*.py

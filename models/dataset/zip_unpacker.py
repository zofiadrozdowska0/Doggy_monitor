import zipfile
import os

# Unpacks all *.zip files in given folder
zip_files_directory = "dir"
output_directory = "2B_14-18"
os.makedirs(output_directory, exist_ok=True)

for item in os.listdir(zip_files_directory):
    if item.endswith('.zip'):
        zip_file_path = os.path.join(zip_files_directory, item)

        extract_to_folder = os.path.join(output_directory, os.path.splitext(item)[0])
        os.makedirs(extract_to_folder, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_folder)
            print(f'Extracted {item} to {extract_to_folder}')

print('All done :)')

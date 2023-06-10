import os
import csv

folder_paths = [
    'HumanGenerated_NewsArticlesID',
    'ngram_ArticlesID',
    'gptID'
]  # Replace with the actual paths to your folders

file_info = []

for folder_path in folder_paths:
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith('.txt'):
            truth_value = '1' if folder_path == 'HumanGenerated_NewsArticlesID' else '0'

            file_info.append({
                'GroupID': 'groupName',
                'FileID': filename,
                'Class': truth_value
            })

            print(f'Added {filename} to file information')

# Create a CSV file and write the file information
csv_file = 'file_info.csv'
fieldnames = ['GroupID', 'FileID', 'Class']

# Check if the CSV file already exists
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write header only if the file didn't exist
    if not file_exists:
        writer.writeheader()

    writer.writerows(file_info)

print(f'File information saved to {csv_file}.')

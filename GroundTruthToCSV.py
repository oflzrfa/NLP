import os
import csv

folder_paths = [
    'HumanGenerated_NewsArticlesID',
    'ngram_ArticlesID',
    'gptID'
]  # Replace with the actual paths to your folders

# Create a list to store file information
file_info = []

# Iterate over the folders
for folder_path in folder_paths:
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a text document
        if filename.endswith('.txt'):
            # Determine the truth value based on the folder
            truth_value = '1' if folder_path == 'HumanGenerated_NewsArticlesID' else '0'

            # Add file information to the list
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

import os
import random
import string

folder_paths = [
    'HumanGenerated_NewsArticlesID',
    'gptID',
    'ngram_ArticlesID'
]  # Replace with the actual paths to your folders

# Function to generate a unique 4-digit number
def generate_unique_number():
    return ''.join(random.choices(string.digits, k=4))

# Iterate over the folder paths
for folder_path in folder_paths:
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is a text document
        if filename.endswith('.txt'):
            # Generate a unique 4-digit number
            new_number = generate_unique_number()
            
            # Create the new filename by combining the number and the file extension
            new_filename = new_number + '.txt'
            
            # Handle filename conflicts by appending a suffix
            count = 1
            while os.path.exists(os.path.join(folder_path, new_filename)):
                new_filename = f'{new_number}_{count}.txt'
                count += 1
            
            # Rename the file
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(file_path, new_file_path)
            
            print(f'Renamed {filename} to {new_filename}')


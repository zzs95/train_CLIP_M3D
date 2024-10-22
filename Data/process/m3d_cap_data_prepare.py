import os  # Module to handle file paths and directories
import numpy as np  # Library for numerical operations and handling arrays
from PIL import Image  # Python Imaging Library (PIL) to handle image files
import concurrent.futures  # Module for parallel processing
from tqdm import tqdm  # For displaying progress bars
from collections import Counter  # To count the occurrences of elements
import unicodedata  # Module to handle Unicode characters
import monai.transforms as mtf  # MONAI is a framework for medical image analysis
from multiprocessing import Pool  # For parallel processing using multiple CPU cores
from unidecode import unidecode  # For converting Unicode characters to ASCII

# Define the input and output directories
# input_dir = 'PATH/M3D_Cap/ct_quizze/'
# output_dir = 'PATH/M3D_Cap_npy/ct_quizze/'

input_dir = 'PATH/M3D_Cap/ct_case/'  # Directory containing input images and text files
output_dir = 'PATH/M3D_Cap_npy/ct_case/'  # Directory where processed data will be saved

# Get all the subfolders in the input directory (e.g., [00001, 00002...])
subfolders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

# Define the image transformation pipeline
transform = mtf.Compose([
    mtf.CropForeground(),  # Removes unnecessary background from images
    mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")  # Resizes images to a fixed size of 32x256x256 pixels
])

# Function to process each subfolder
def process_subfolder(subfolder):
    output_id_folder = os.path.join(output_dir, subfolder)  # Create output folder path for each subfolder
    input_id_folder = os.path.join(input_dir, subfolder)  # Create input folder path for each subfolder

    os.makedirs(output_id_folder, exist_ok=True)  # Create output folder if it doesn't exist

    for subsubfolder in os.listdir(input_id_folder):  # Loop through files in each subfolder
        if subsubfolder.endswith('.txt'):  # If the file is a text file
            text_path = os.path.join(input_dir, subfolder, subsubfolder)  # Full path to the text file
            with open(text_path, 'r') as file:  # Open and read the text file
                text_content = file.read()

            # Look for a specific section of the text (e.g., "study_findings:")
            search_text = "study_findings:"
            index = text_content.find(search_text)

            # If the section is found, extract the text after it
            if index != -1:
                filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip()
            else:
                print("Specified string not found")  # Print a warning if not found
                filtered_text = text_content.replace("\n", " ").strip()  # Remove newlines and extra spaces

            # If the text is too short, try looking for a different section (e.g., "discussion:")
            if len(filtered_text.replace("\n", "").replace(" ", "")) < 5:
                search_text = "discussion:"
                index = text_content.find(search_text)
                if index != -1:
                    filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip()
                else:
                    print("Specified string not found")
                    filtered_text = text_content.replace("\n", " ").strip()

            # Save the filtered text to a new file in the output directory
            new_text_path = os.path.join(output_dir, subfolder, subsubfolder)
            with open(new_text_path, 'w') as new_file:
                new_file.write(filtered_text)

        # Handle image subfolders
        subsubfolder_path = os.path.join(input_dir, subfolder, subsubfolder)

        if os.path.isdir(subsubfolder_path):  # If it's a directory (i.e., contains images)
            subsubfolder = unidecode(subsubfolder)  # Normalize the folder name (e.g., convert "Pöschl" to "Poschl")
            output_path = os.path.join(output_dir, subfolder, f'{subsubfolder}.npy')  # Path for the output file

            # Get all image files with '.jpeg' or '.png' extensions
            image_files = [file for file in os.listdir(subsubfolder_path) if
                           file.endswith('.jpeg') or file.endswith('.png')]

            if len(image_files) == 0:  # Skip if no images are found
                continue

            # Sort images based on their filenames (assuming filenames are numbers)
            image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

            images_3d = []  # List to hold 3D image stacks
            for image_file in image_files:
                image_path = os.path.join(subsubfolder_path, image_file)  # Full path to each image
                try:
                    img = Image.open(image_path)  # Open the image
                    img = img.convert("L")  # Convert image to grayscale
                    img_array = np.array(img)  # Convert image to a numpy array
                    # Normalize pixel values to the range [0, 1]
                    img_array = img_array.astype(np.float32) / 255.0
                    images_3d.append(img_array[None])  # Add the image to the 3D stack
                except:
                    print("This image is error: ", image_path)  # Print a message if an error occurs

            images_3d_pure = []  # List for storing correctly shaped images
            try:
                img_shapes = [img.shape for img in images_3d]  # Get the shapes of all images
                item_counts = Counter(img_shapes)  # Count occurrences of each shape
                most_common_shape = item_counts.most_common(1)[0][0]  # Get the most common shape
                for img in images_3d:
                    if img.shape == most_common_shape:  # Only keep images with the most common shape
                        images_3d_pure.append(img)
                final_3d_image = np.vstack(images_3d_pure)  # Stack the images into a 3D array

                image = final_3d_image[np.newaxis, ...]  # Add a new dimension for the 3D image

                # Normalize the image data
                image = image - image.min()
                image = image / np.clip(image.max(), a_min=1e-8, a_max=None)

                img_trans = transform(image)  # Apply the transformation pipeline

                np.save(output_path, img_trans)  # Save the transformed image as a numpy file (.npy)
            except:
                print([img.shape for img in images_3d])  # Print the shapes of the images if an error occurs
                print("This folder is vstack error: ", output_path)  # Print an error message

# Use multiprocessing to process all subfolders in parallel using 32 CPU cores
with Pool(processes=32) as pool:
    with tqdm(total=len(subfolders), desc="Processing") as pbar:  # Display a progress bar
        for _ in pool.imap_unordered(process_subfolder, subfolders):  # Process subfolders in parallel
            pbar.update(1)  # Update the progress bar after each subfolder is processed

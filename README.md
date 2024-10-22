Here is the tutorial for training a CLIP model on your own 3D medical image data, using M3D source code as an example.

# Training a CLIP Model Using 3D Medical Image Data

This guide explains how to train a CLIP model using 3D medical images and corresponding textual reports. We use the M3D data format for preprocessing, where each sample consists of an image and a text pair. The preprocessing steps are inspired by the [M3D caption dataset example](https://huggingface.co/datasets/GoodBaiBai88/M3D-Cap/tree/main/data_examples) and the [data preparation code](https://github.com/zzs95/train_CLIP_M3D/tree/main/Data/data/examples).

### Data Structure Example
For each sample, images and text are stored together in a folder:

```plaintext
data_folder
-- 000007
   |- img.npy
   |- text.txt
-- 000008
   |- img.npy
   |- text.txt
-- 000009
   |- img.npy
   |- text.txt
-- 000010
   |- img.npy
   |- text.txt
```

## 1. Data Preprocessing
Using code of M3D caption data preprocess as an example. [m3d_cap_data_prepare](https://github.com/zzs95/train_CLIP_M3D/blob/main/Data/process/m3d_cap_data_prepare.py)
### a. Image Data Preprocessing

M3D uses fixed resolution images with a shape of `(32, 256, 256)` for the model, meaning we need to preprocess the original DICOM or NIfTI images and convert them into `.npy` format. The preprocessing pipeline involves several key steps:

1. **Spatial Correction**: Ensure the images are spatially aligned. This includes adjusting the image direction and spacing, important in medical imaging to standardize across different datasets.
   
2. **Crop to Region of Interest**: Focus the model's attention by cropping the image to the relevant anatomical area.

3. **Resizing with Cropping and Padding**: The 3D images should be resized to a resolution of `(32, 256, 256)` by cropping or padding the images to this size, ensuring uniform input dimensions.

4. **Intensity Normalization**: Depending on the task, normalize the pixel values (either grayscale or Hounsfield Units (HU)) to remove variability caused by different imaging protocols or scanners.

5. **Save as `.npy`**: Finally, save the preprocessed 3D image as an `.npy` file named `img.npy` in the corresponding folder.

### b. Text Data Preprocessing

1. **Extract Relevant Text**: From the medical report, extract the sections of interest (e.g., "Findings" or "Impression"). Depending on your use case, you can choose to extract just one section or multiple sections, or even the full report.

2. **Save as `text.txt`**: Save the extracted text into a file named `text.txt` in the same folder as the corresponding image.

### c. Creating a Dataset JSON File

After preparing the images and text, you need to organize the dataset by creating a JSON file. This file will point to the image-text pairs for training, validation, and testing.

Example structure for `dataset.json`:

```json
{
    "train": [
        {
            "image": "data_folder/000007/img.npy",
            "text": "data_folder/000007/text.txt"
        },
        {
            "image": "data_folder/000008/img.npy",
            "text": "data_folder/000008/text.txt"
        }
    ],
    "validation": [
        {
            "image": "data_folder/000009/img.npy",
            "text": "data_folder/000009/text.txt"
        }
    ],
    "test": [
        {
            "image": "data_folder/000010/img.npy",
            "text": "data_folder/000010/text.txt"
        }
    ]
}
```

## 2. Training the CLIP Model

### a. Modifying the Dataset Loader

To load the image-text pairs for training the CLIP model, you need to modify the dataset loader. An example of a dataset class can be found in [CapDataset](https://github.com/zzs95/train_CLIP_M3D/blob/782c8f7c673d6167efbe753d57ef635842b7d302/LaMed/src/dataset/multi_dataset.py#L132). This class reads `.npy` files for images and corresponding `.txt` files for text. Adjust the file paths to match your dataset's directory structure.

### b. Adjusting Training Script Parameters

To train the CLIP model, you'll need to update the training script's parameters. Specifically:

- **data_root**: The root directory of your data.
- **cap_data_path**: Path to the JSON file that contains the image-text pairs.
- **batch_size**: The batch size used for training, which can depend on your GPU's memory.

Refer to the CLIP training script [train_CLIP.py](https://github.com/zzs95/train_CLIP_M3D/blob/main/LaMed/src/train/train_CLIP.py) to adjust these settings.

### c. Optional: Modifying Pretrained Models

You can also experiment with different pretrained models for both the image and text encoders. This might involve choosing a pretrained vision model (e.g., ResNet, Vision Transformer) or using a different text encoder (e.g., BERT). Modifying these components allows for more tailored representations, especially for domain-specific data like medical images.

## 3. Training Workflow

Once the dataset and training parameters are set, the CLIP training process proceeds as follows:

1. **Data Loading**: The dataset is loaded from the JSON file, pairing 3D medical images with their corresponding reports.
2. **Model Training**: The CLIP model learns to associate the image and text representations in a joint embedding space.
3. **Evaluation**: The model is evaluated on the validation and test sets to measure its performance on unseen data.

By following these steps, you can train a CLIP model on your own 3D medical image data, enabling it to generate meaningful relationships between complex medical images and corresponding textual reports.

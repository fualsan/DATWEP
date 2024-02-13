# DATWEP
Source code for the **Dynamic Task and Weight Prioritization Curriculum Learning for Multimodal Imagery** paper (currently under review). 

## Setup
Create a new Python environment

```console
$ conda create -n datwep python=3.11 -y
```
Activate created environment

```console
$ conda activate datwep
```

Install dependencies
```console
$ pip install -r requirements.txt
```


## Downloading & Preparing dataset
Download the FloodNet dataset from below (Both Track 1 and 2):
* https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021

1. Download track 1 & 2 from https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021
2. Extract “Images” and “Questions” folders from track 2 files
3. Extract training images from track 1 files. Since validation and test images don’t have annotations, we are going to ignore them.
4. In “labeled” folder, combine mask folders from “Flooded” and “Non-Flooded” folders in a new directory. 
5. Create three new directories and move files: 
    1. “track2_vqa/Images/” move “Images” folder from track 2 files
    2. “track2_vqa/Questions/” move “Questions” folder from track 2 files
    3. “track1_seg/train-label-img” move combined mask files here (mask folders from “Flooded” and “Non-Flooded” folders)
6. If your prefer other directories, please modify the directories in hyperparameters dictionary:
    1. hyperparameters['DATASET']['IMAGES_ROOT']
    2. hyperparameters['DATASET']['QUESTIONS_ROOT']
    3. hyperparameters['DATASET']['MASK_IMAGES_PATH']
7. Note: We only use the “Train_Image” folder of the “Images” of track 2. This is automatically set.
8. If you prefer another directory, then you should set the **DATA_ROOT** in Training.ipynb notebook.

```python
hyperparameters['DATASET']['DATA_ROOT']
```

## Experiments
1. After setting up the dataset, run the jupyter server locally:

```console
$ jupyter lab
```

2. Open **Training.ipynb** notebook, make sure correct data folders are set. This notebook will train the model and save the results. 

3. After training is complete, open **Evaluation.ipynb** notebook and run to see the final results.

## Contact
If you have any questions, feel free the contact us:

H. Fuat Alsan (PhD Candidate)
huseyinfuat.alsan@stu.khas.edu.tr

Assoc. Prof. Dr. Taner Arsan (Computer Engineering Department Chair)
arsan@khas.edu.tr


## If you use our work, please cite us:
(PAPER STILL UNDER REVIEW, WILL BE UPDATED LATER)

## Prediction Examples
#### VQA Example Prediction
![Example VQA Prediction](assets/vqa_pair_1.jpg)

### Segmentation Example 1
#### Acutal Image
![Example Segmentation Image 1](assets/segmentation_image_1.jpg)

#### Acutal Masks
![Example Segmentation Masks Actual 1](assets/masks_actual_1.jpg)

#### Predicted Masks
![Example Segmentation Masks Prediction 1](assets/masks_prediction_1.jpg)

#### Acutal Masks (Combined Color)
![Colorful Segmentation Masks Actual 1](assets/colorful_masks_actual_1.jpg)

#### Predicted Masks (Combined Color)
![Colorful Segmentation Masks Prediction 1](assets/colorful_masks_prediction_1.jpg)

### Segmentation Example 2
#### Acutal Image
![Example Segmentation Image 2](assets/segmentation_image_2.jpg)

#### Acutal Masks
![Example Segmentation Masks Actual 2](assets/masks_actual_2.jpg)

#### Predicted Masks
![Example Segmentation Masks Prediction 2](assets/masks_prediction_2.jpg)

#### Acutal Masks (Combined Color)
![Colorful Segmentation Masks Actual 2](assets/colorful_masks_actual_2.jpg)

#### Predicted Masks (Combined Color)
![Colorful Segmentation Masks Prediction 2](assets/colorful_masks_prediction_2.jpg)

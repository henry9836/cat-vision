
Cat Vision 0.5 - v1 2023-10-11 10:53pm
==============================

This dataset was exported via roboflow.com on October 11, 2023 at 9:55 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 809 images.
Cucumber-snake are annotated in folder format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -45 and +45 degrees
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Salt and pepper noise was applied to 5 percent of pixels



# Cat and Non-Cat Classification
# Dependencies and helper functions

# Dependencies
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import imghdr

# Avoid OOM errors by setting GPU memory consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# Helper functions
def remove_invalid_images(data_dir, image_extensions):
    """
    Remove invalid images from the given directory.

    Args:
        data_dir (str): The directory containing the images.
        image_extensions (list): List of valid image extensions.

    Returns:
        None
    """
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_extensions:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))
                # os.remove(image_path)

# Example usage
data_dir = 'C:\\Users\\User\\Downloads\\Python Class\\myenv\\binary_classification\\data'
image_extensions = ['jpg', 'jpeg', 'png']
remove_invalid_images(data_dir, image_extensions)

# Load an image
#img = cv2.imread('C:\\Users\\User\\Downloads\\Python Class\\myenv\\binary_classification\\data\\cat\\8TzrzMKac.jpg')    
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()

# load data
# data loader
data = tf.keras.utils.image_dataset_from_directory(data_dir)

# get the first batch of data
data_iterator = data.as_numpy_iterator()

# get a batch of data
# class 1 is dog, class 0 is cat
batch = data_iterator.next()

# plot the first 4 images in the batch
fig, ax = plt.subplots(ncols = 4, figsize=(20, 20))
for i, img in enumerate(batch[0][:4]):
    ax[i].imshow(img.astype(int))
    ax[i].set_title(data.class_names[batch[1][i]])
    ax[i].axis('off')
plt.show()
print(batch[1][:4])






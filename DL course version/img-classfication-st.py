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

def load_dataset(data_dir):
    """
    Load the dataset from the given directory.

    Args:
        data_dir (str): The directory containing the dataset.

    Returns:
        train_set_x_orig (numpy.ndarray): Original training set images.
        train_set_y (numpy.ndarray): Training set labels.
        test_set_x_orig (numpy.ndarray): Original test set images.
        test_set_y (numpy.ndarray): Test set labels.
        classes (list): List of classes (in this case, ['cat', 'non-cat']).
    """
    # Example usage
    image_extensions = ['jpg', 'jpeg', 'png']
    remove_invalid_images(data_dir, image_extensions)
    
    # Load an image
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    
    # get the first batch of data
    data_iterator = data.as_numpy_iterator()
    
    # get a batch of data
    # class 1 is dog, class 0 is cat
    batch = data_iterator.next()
    
    train_set_x_orig = batch[0]
    train_set_y = batch[1]
    
    # Assuming a 70-30 train-test split
    test_set_x_orig = data_iterator.next()[0]
    test_set_y = data_iterator.next()[1]
    
    classes = data.class_names
    
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes

# Example usage
data_dir = 'C:\\Users\\User\\Downloads\\Python Class\\myenv\\binary_classification\\data'
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(data_dir)

# Plot the first 4 images in the training set
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for i, img in enumerate(train_set_x_orig[:4]):
    ax[i].imshow(img.astype(int))
    ax[i].set_title(classes[train_set_y[i]])
    ax[i].axis('off')
plt.show()
print(train_set_y[:4])

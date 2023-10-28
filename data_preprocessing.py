from sklearn.model_selection import train_test_split
import os
from PIL import Image
from itertools import cycle, islice
import numpy as np
from tensorflow import keras

load_img = keras.preprocessing.image.load_img
img_to_array = keras.preprocessing.image.img_to_array

"""
80-20 Split

Train: Animation[] (list of Animation sequences)
Target: ShiftedAnimation[]
Test: Frame[]

Animation: Frame[] x 15 (from image 1-15)
ShiftedAnimation: Frame[] x 15 (from image 2-16)
Frame: Datatype (128,128,3)
"""
IMG_SIZE = 128
# Get all the data inputs in Train-Data by folders into a List of List of Image Filepaths
def preprocess():
    def get_animation_filepaths(directory_path):
        # Initialize an empty list to store subdirectories and their files
        animation_list = []

        # Loop through each entry in the directory
        for entry in os.listdir(directory_path):
            entry_path = os.path.join(directory_path, entry)

            # Check if the entry is a subdirectory
            if os.path.isdir(entry_path):
                # Get the list of files in the subdirectory
                frames = os.listdir(entry_path)
                frames.sort(key=lambda x: int(x.split('.')[0]))
                frames = [f"{entry_path}/{frame}" for frame in frames]
                # Add the subdirectory and its files to the list
                animation_list.append(frames)

        return animation_list

    # Replace 'path/to/your/directory' with the actual directory path
    directory_path = 'Train-Data'
    animation_filepaths = get_animation_filepaths(directory_path)
    #print(animation_filepaths)

    # for each Image Filepath, load it as a (128, 128, 3) image and extend it to 15 frames

    def get_animations(animation_filepaths):
        animations = []
        for animation_filepath in animation_filepaths:
            animation = []
            
            # Load frame into (128, 128, 3)
            for frame_file in animation_filepath:
                #img = Image.open(frame_file).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                img = load_img(frame_file, target_size=(128, 128), color_mode='rgb')
                img_array = img_to_array(img)
                animation.append(img_array)
                
            # Extend Frames to 16
            target_size = 16
            # Use itertools.cycle to create an iterator that repeats the original array
            repeated_frames = islice(cycle(animation), target_size)
            animations.append(list(repeated_frames))
        return np.array(animations)

    animations = get_animations(animation_filepaths)
    print("Full dataset size: ", animations.shape)


    # Use a train-Test split on the Animation List
    test_size = 0.2
    train_anims, test_anims = train_test_split(animations, test_size=test_size, random_state=42)

    # Created a ShiftedAnimation on Train, where each Animation frame is shifted circularly to the right by one
    def train_target_split(animations):
        train = []
        target = []
        for anim in animations:
            train.append(anim[:-1,:,:,:])
            target.append(anim[1:,:,:,:])
        return np.array(train), np.array(target)

    train_anims, target_anims = train_target_split(train_anims)
    print("Train dataset size: ", train_anims.shape, "Target dataset size: ", target_anims.shape, "Test dataset size: ", test_anims.shape)
    return train_anims, target_anims, test_anims

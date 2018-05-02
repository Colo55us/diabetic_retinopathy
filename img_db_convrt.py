import time

import numpy as np
import pandas as pd
from PIL import Image



def convert_images_to_arrays_train(file_path, df):
    
    lst_imgs = [l for l in df['image']]

    return np.array([np.array(Image.open(file_path + img)) for img in lst_imgs],dtype='float32')


def save_to_array(arr_name, arr_object):
    
    return np.save(arr_name, arr_object)


if __name__ == '__main__':
 

    labels = pd.read_csv("data/csv/final.csv")
    X_train = convert_images_to_arrays_train('data/train_resized/', labels)
    print(X_train.shape)
    save_to_array('data/training_data.npy', X_train)


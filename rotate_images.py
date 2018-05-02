import pandas as pd
import numpy as np
from skimage import io
from skimage.transform import rotate
import cv2
import os
import time

def rotate_images(file_path, degrees_of_rotation, file_name):

    #for l in lst_imgs:
    img = io.imread(file_path + str(file_name) )
    img = rotate(img, degrees_of_rotation)
    io.imsave('data/rotated_images/' + 'rot_' + str(file_name), img)
    return 'rot_' + str(file_name)

def mirror_images(file_path, mirror_direction, file_name):
    
    
    img = cv2.imread(file_path + str(file_name))
    img = cv2.flip(img, 1)
    cv2.imwrite('data/mirrored_images/' + 'mir_' + str(file_name) , img)
    return 'mir_' + str(file_name)

if __name__ == '__main__':
    '''
    
    trainLabels = pd.read_csv("trainLabels_master.csv")

    trainLabels['image'] = trainLabels['image'].str.rstrip('.jpeg')
    trainLabels_no_DR = trainLabels[trainLabels['level'] == 0]
    trainLabels_DR = trainLabels[trainLabels['level'] >= 1]

    lst_imgs_no_DR = [i for i in trainLabels_no_DR['image']]
    lst_imgs_DR = [i for i in trainLabels_DR['image']]

   

    # Mirror Images with no DR one time
    print("Mirroring Non-DR Images")
    mirror_images('../data/train-resized-256/', 1, lst_imgs_DR)


    # Rotate all images that have any level of DR
    print("Rotating 90 Degrees")
    rotate_images('../data/train-resized-256/', 90, lst_imgs_DR)
    '''
    start_time = time.time()
    df = pd.read_csv('rotated_images.csv')
    labels = pd.read_csv('trainLabels_master.csv')
    count = 1
    for i,row in labels.iterrows():
        
        if row['level'] > 0:
            img_name = row['image']
            x = rotate_images('data/train_resized/',270,img_name)
            df = df.append({'image':x,'level':1},ignore_index=True)
            print(count)
            count +=1
    df.to_csv('rotated_images.csv',sep=',')

    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))

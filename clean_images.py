import time
import numpy as np
import pandas as pd
from PIL import Image
import os
from skimage import io
def find_black_images(file_path, df):
   
    result = []
    lst_imgs = [l for l in df['image']]
    for img in lst_imgs:
        if np.mean(np.array(Image.open(file_path+img))) == 0:
            result.append(1)
            os.remove(file_path+img)
        else:
            result.append(0)
            file = Image.open(file_path+img)
            io.imsave(str('data/train_resized_cleaned' + img), file)

    return result


if __name__ == '__main__':
    start_time = time.time()
    trainLabels = pd.read_csv('check.csv')

    trainLabels['image'] = [i + '.jpeg' for i in trainLabels['image']]
    trainLabels['black'] = np.nan

    trainLabels['black'] = find_black_images('data/train_resized/', trainLabels)
    trainLabels = trainLabels.loc[trainLabels['black'] == 0]
    trainLabels.to_csv('trainLabels_master.csv', index=False, header=True)

    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))

import numpy as np
import pandas as pd

from keras.layers import Dropout,Flatten,MaxPooling2D,Dense,Activation 
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import precision_score,recall_score




def reshape_data(arr, img_rows, img_cols, channels):
   
    return arr.reshape(arr.shape[0], img_rows, img_cols, channels)


def convnet(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes):
    

    model = Sequential()

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     strides=1,
                     input_shape=(img_rows, img_cols, channels), activation="relu"))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)

    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

   

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,verbose=1,validation_split=0.2,class_weight='auto')
    return model
    


if __name__ == '__main__':
    
    batch_size = 16 # suitable gor gpu with 6 gigs of vram and if you have increade it
    nb_classes = 2 
    nb_epoch = 25

    img_rows, img_cols = 256, 256
    channels = 3 # 3 for rgb .If you are using grayscale replace it with 1
    nb_filters = 32
    kernel_size = (8, 8)


    labels = pd.read_csv("data/csv/final.csv")
    X = np.load("data/training_data.npy")
    y = np.array(labels['level'])
  

    print("Preparing data for modelling....")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    X_train = reshape_data(X_train, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    input_shape = (img_rows, img_cols, channels)
    '''
    # cant normalize due to memory storage getting full 
    # if you have atleast 16Gb of ram then remove the triple qoutes
    #X_train = X_train.astype('float32')
    
    X_test = X_test.astype('float32')
  
    X_train /= 255
    X_test /= 255
    '''
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)
    print('||Data preparation complete||')
    print('Model training starts')
    

    model = cnn_model(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size,
                      nb_classes)

    
    y_pred = model.predict(X_test)

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Precision: ", precision)
    print("Recall: ", recall)
    model.save("convnet_dr.h5")
    
    print(" Model training Completed and model stored in cwd")

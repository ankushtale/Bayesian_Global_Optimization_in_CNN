from scipy.io import loadmat
x = loadmat('svhn_data/train_32x32.mat')
y = loadmat('svhn_data/test_32x32.mat')
import gzip
import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
import tensorflow
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense

def create_model(lr = 0.001, decay = 0.0001,DROPOUT_P1 = 0.2,DROPOUT_P2 = 0.2 ):
    
#     model = Sequential()
#     model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
#     model.add(layers.AveragePooling2D())
#    
#     model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
#     model.add(layers.AveragePooling2D())
#     model.add(layers.Flatten())
#    
#     model.add(layers.Dense(units=120, activation='relu'))
#     model.add(layers.Dropout(rate=DROPOUT_P1, seed=1))
#     model.add(layers.Dense(units=84, activation='relu'))
#     model.add(layers.Dropout(rate=DROPOUT_P2, seed=1))
#     model.add(layers.Dense(units=1, activation = 'softmax'))
#     #, kernel_initializer="glorot_uniform", bias_initializer="zeros")
#     model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(lr=lr,decay=decay), metrics=['accuracy'])
     model = keras.Sequential()
     model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(32,32,3), strides=1, padding='same', kernel_initializer="he_uniform", bias_initializer="zeros"))
     #model.add(layers.AveragePooling2D())
    
     model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', kernel_initializer="he_uniform", bias_initializer="zeros"))
     #model.add(layers.AveragePooling2D())
     model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=1, padding='same', kernel_initializer="he_uniform", bias_initializer="zeros"))
    
     model.add(layers.Flatten())
    
     model.add(layers.Dense(units=1024, activation='relu', kernel_initializer="glorot_uniform", bias_initializer="zeros"))
     model.add(layers.Dropout(rate=DROPOUT_P1, seed=1))
     model.add(layers.Dense(units=1024, activation='relu', kernel_initializer="glorot_uniform", bias_initializer="zeros"))
     model.add(layers.Dropout(rate=DROPOUT_P2, seed=1))
     model.add(layers.Dense(units=10, activation = 'softmax'))
     model.compile(loss="categorical_crossentropy", optimizer=tensorflow.keras.optimizers.Adam(lr=lr,decay=decay), metrics=['accuracy'])
     return model

train_labels = x['y']


#[3]:


train_features = np.swapaxes(np.swapaxes(np.swapaxes(x['X'],2,3), 1,2), 0,1)


# [4]:


test_labels = y['y']


# [5]:


test_features = np.swapaxes(np.swapaxes(np.swapaxes(y['X'],2,3), 1,2), 0,1)
train_labels_count = np.unique(train_labels, return_counts=True)
dataframe_train_labels = pd.DataFrame({'Label':train_labels_count[0], 'Count':train_labels_count[1]})
dataframe_train_labels


# ## Train Test Split



validation = {}
train_features, validation_features, train_labels, validation_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=0)




#train_labels.shape




print('# of training images:', train_features.shape[0])
print('# of validation images:', validation_features.shape[0])

#def create_model(lr = 0.001, decay = 0.0001,DROPOUT_P1 = 0.2,DROPOUT_P2 = 0.2 ):
#    
#     model = Sequential()
#     model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
#     model.add(layers.AveragePooling2D())
#    
#     model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
#     model.add(layers.AveragePooling2D())
#     model.add(layers.Flatten())
#    
#     model.add(layers.Dense(units=120, activation='relu'))
#     model.add(layers.Dropout(rate=DROPOUT_P1, seed=1))
#     model.add(layers.Dense(units=84, activation='relu'))
#     model.add(layers.Dropout(rate=DROPOUT_P2, seed=1))
#     model.add(layers.Dense(units=10, activation = 'softmax'))
#     #, kernel_initializer="glorot_uniform", bias_initializer="zeros")
#     model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(lr=lr,decay=decay), metrics=['accuracy'])
#     return model

      

X_train, y_train = train_features,to_categorical(train_labels)[:,1:]
# X_train = train_features

X_validation, y_validation = validation_features, to_categorical(validation_labels)[:,1:]

lr = list(np.logspace(np.log(0.00001), np.log(0.01), num = 100, base=3))
decay = list(np.logspace(np.log(0.0001), np.log(0.1), num = 10, base=2.73))
batch_size = [16,32,64,128,256]
DROPOUT_P2 = np.random.uniform(0.1,0.7,size=(10,))
DROPOUT_P1 = np.random.uniform(0.1,0.7,size=(10,))

#param_random = {'lr' : lr, 'decay' : decay,'batch_size' : batch_size,'DROPOUT_P1' : DROPOUT_P1,'DROPOUT_P2' : DROPOUT_P2, 'batch_size' : batch_size}
param_random = {'lr' : lr}

model = KerasClassifier(build_fn = create_model,verbose=1)
random = RandomizedSearchCV(estimator=model, cv=KFold(2), param_distributions=param_random, 
                          verbose=20,  n_iter=30, n_jobs=1,scoring = 'accuracy')

random_fit= random.fit(X_train, y_train)
random_results = pd.DataFrame(random_fit.cv_results_)
random_results.to_csv("random_search_results.csv")
print(random_fit.best_score_)
print(random_fit.best_estimator_)

############# Grid Search#############

lr=[1e-2, 1e-3, 1e-4]
decay=[1e-6,1e-9,0]
dropout1 = [0, 0.1, 0.2, 0.3]
dropout2 = [0,0.1,0.2,0.3]
param_grid = dict(lr = lr, decay = decay, batch_size = batch_size,DROPOUT_P1 = dropout1, DROPOUT_P2 = dropout2)

grid = GridSearchCV(estimator=model, cv=KFold(3), param_distributions=param_grid, 
                          verbose=20,  n_iter=10, n_jobs=1,scoring = 'loss')

grid_fit = grid.fit(train_features, train_labels)

grid_result = pd.DataFrame(grid.cv_results_)
grid_result.to_csv('gridsearch_result.csv')
print(grid_result.best_params_)

best_model = grid_result.best_estimator_
#
#lr = list(np.logspace(np.log(0.01), np.log(1), num = 10, base=3))
#lr_decay = list(np.logspace(np.log(0.0001), np.log(0.1), num = 10, base=2.73))
#batch_size = [16,32,64,128,256]
#dropout = np.random.uniform(0.1,0.7,size=(10,))
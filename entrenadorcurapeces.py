import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import sys
import zipfile
import tarfile
import sys
import os
import math
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
#librerias


#varibles globales

K.clear_session()
"""solo si es para  la enefemedad"""
# especimendepractica = './especimendepractica/peces/enfermos/'
# especimendeexamen= './especimendeexamen/peces/enfermos/'
"""solo si es para  la sie eta sano"""
especimendepractica = './especimendepractica/peces/'
especimendeexamen= './especimendeexamen/peces/'
"""solo si es para  si es un pez"""
#especimendepractica = './especimendepractica'
#especimendeexamen= './especimendeexamen'
pruebas = 1
alturadelaimagen =150
longituddelaimagen= 150
numerodeimagenesamandar=64
veces =500#numero de veces que se va aprosesar la informacion
validacion= veces/math.e
filtroprimeravez= 32
filtrosegundavez= 64
filtroterceravez= 32
filtrocurtavez =128
filtroquintavez =256
filtrouno=(3,3)
filtrodos=(2,2)
filtrotres=(3,3)
filtrocutro=(4,4)
filtroquinto=(5,5)
pulido=(2,2)
numerodenfermedades=2# cambiar mientras encuenbtro imagenes y la sano cuenta como enfermedad
#numerodenfermedades=2
lr = 0.00004



entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    especimendepractica,
    target_size=(alturadelaimagen, longituddelaimagen),
    batch_size=numerodeimagenesamandar,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    especimendeexamen,
    target_size=(alturadelaimagen, longituddelaimagen),
    batch_size=numerodeimagenesamandar,
    class_mode='categorical')

cnn = Sequential()
cnn.add(Convolution2D(filtroprimeravez, filtrouno, padding ="same", input_shape=(longituddelaimagen, alturadelaimagen , 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=pulido))

cnn.add(Convolution2D(filtrosegundavez, filtrodos, padding ="same"))
cnn.add(MaxPooling2D(pool_size=pulido))

# cnn.add(Convolution2D(filtroterceravez, filtrotres, padding ="same"))
# cnn.add(MaxPooling2D(pool_size=pulido))
#
# cnn.add(Convolution2D(filtrocurtavez, filtrocutro, padding ="same"))
# cnn.add(MaxPooling2D(pool_size=pulido))
#
# cnn.add(Convolution2D(filtroquintavez, filtroquinto, padding ="same"))
# cnn.add(MaxPooling2D(pool_size=pulido))

cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(numerodenfermedades, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])




cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=veces,
    epochs=pruebas,
    validation_data=validacion_generador,
    validation_steps=validacion)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

#cnn.save('./modelo/modelo.h5')
# #cnn.save_weights('./modelo/pesos.h5')
cnn.save('./modelo/modelopezenfermo.h5')
cnn.save_weights('./modelo/pesospezenfermo.h5')
# cnn.save('./modelo/modelopez.h5')
# cnn.save_weights('./modelo/pesospez.h5')

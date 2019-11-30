import numpy as np
import tensorflow as tf
import cv2
import sys
import zipfile
import tarfile
import os
#import pycuda.driver as cuda
#import pycuda.autoinit
#from pycuda.compiler import SourceModule
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow.python.keras import backend as K
#from numba import cuda
#import numba
import orendador

#librerias


#varibles globales
#@jit(target ="cuda")
#def curapeces():
class curapeces():
    K.clear_session()
    """solo si es para  la enefemedad"""
    #especimendepractica = './especimendepractica/peces/enfermos/'
    #especimendeexamen= './especimendeexamen/peces/enfermos/'
    """solo si es para  la sie eta sano"""
    #especimendepractica = './especimendepractica/peces/'
    #especimendeexamen= './especimendeexamen/peces/'
    """solo si es para  si es un pez"""
    #especimendepractica = './especimendepractica'
    #especimendeexamen= './especimendeexamen'
    directorio="./datos_limpios"
    pruebas = 1
    alturadelaimagen =150
    longituddelaimagen= 150
    numerodeimagenesamandar=2
    pasos=100#numero de veces que se va aprosesar la informacion
    validacon=183
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
    numerodenfermedades=14# cambiar mientras encuenbtro imagenes y la sano cuenta como enfermedad
    #numerodenfermedades=2
    lr = 0.00004


    def image(self):
        self.entrenamiento_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.entrenamiento_generador = self.entrenamiento_datagen.flow_from_directory(
            self.directorio,
            target_size=(self.alturadelaimagen, self.longituddelaimagen),
            batch_size=self.numerodeimagenesamandar,
            class_mode='categorical')

        self.validacion_generador = self.test_datagen.flow_from_directory(
            self.directorio,
            target_size=(self.alturadelaimagen, self.longituddelaimagen),
            batch_size=self.numerodeimagenesamandar,
            class_mode='categorical')
    def nn(self):
    
        nn = Sequential()
        nn.add(Convolution2D(self.filtroprimeravez, self.filtrouno, padding ="same", input_shape=(self.longituddelaimagen, self.alturadelaimagen , 3), activation='relu'))
        nn.add(MaxPooling2D(pool_size=self.pulido))

        nn.add(Convolution2D(self.filtrosegundavez, self.filtrodos, padding ="same"))
        nn.add(MaxPooling2D(pool_size=self.pulido))

        # nn.add(Convolution2D(self.filtroterceravez, self.filtrotres, padding ="same"))
        # nn.add(MaxPooling2D(pool_size=self.pulido))
        #
        # nn.add(Convolution2D(self.filtrocurtavez, self.filtrocutro, padding ="same"))
        # nn.add(MaxPooling2D(pool_size=self.pulido))
        #
        # nn.add(Convolution2D(filtroquintavez, filtroquinto, padding ="same"))
        # nn.add(MaxPooling2D(pool_size=pulido))

        nn.add(Flatten())
        nn.add(Dense(512, activation='relu'))
        nn.add(Dropout(0.5))
        nn.add(Dense(self.numerodenfermedades, activation='softmax'))
        nn.compile(optimizer='sgd', loss='mse')
        #cnn.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])




        nn.fit_generator(self.entrenamiento_generador,steps_per_epoch=self.pasos,epochs=self.pruebas,validation_data=self.validacion_generador,validation_steps=self.validacon)
        #cnn.save('./modelo_lab_experimental/modelo_pezenfermo.h5')
        #cnn.save_weights('./modelo_lab_experimental/pesospezenfermo.h5')
        return nn
    def save_nn(self):
        self.nn=curapeces.nn()
        self.target_dir = orendador.archivo_existe.archivo_existe()
        self.nn.save(self.target_dir+ '/model.h5')
        self.nn.save_weights(self.target_dir +'/weights.h5')
#cnn.save('./modelo_lab_experimental/modelo_pez.h5')
#cnn.save_weights('./modelo_lab_experimental/pesospez.h5')
#cuarpeces = cuda.mem_alloc(curapecs())
#cuda.memcpy_htod(curapecs, curapecs())
#curapeces()
class predict():
    pez="./para_ensallar/"+input()
    imagenpez = cv2.imread(pez, cv2.IMREAD_COLOR)
    numfolders=len(os.listdir("modelos_de_inteligencia_artificial_variedad"))-1
    modelfolder="./modelos_de_inteligencia_artificial_variedad/curapeces"+str(numfolders)+"__models curapeces__2019-11-29"
    model=modelfolder+"/model.h5"
    weights=modelfolder+"/weights.h5"
    longitud, altura = 150, 150
    def display_image(self):
        cv2.imshow ('ventana1',self.imagenpez)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def predict(self):
      self.nn =  tf.keras.models.load_model(self.model)
      self.nn.load_weights(self.weights)
      self.x = load_img(self.pez, target_size=(self.longitud, self.altura))
      self.x = img_to_array(self.x)
      self.x = np.expand_dims(self.x, axis=0)
      self.array = self.nn.predict(self.x)
      self.result = self.array[0]
      self.answer = np.argmax(self.result)
      #x2 = load_img(file, target_size=(longitud, altura))
      #x2 = img_to_array(x2)
      #x2 = np.expand_dims(x2, axis=0)
      #array = cnn2.predict(x2)
      #result = array[0]
      #self.answer = np.argmax(result)
      if self.answer == 0:
        print("prediccion:  atcado o tumor y deformidad")
      #x3 = load_img(file, target_size=(longitud, altura))
      #x3 = img_to_array(x3)
      #x3 = np.expand_dims(x3, axis=0)
      #array = cnn3.predict(x3)
      #result = array[0]
      #self.answer = np.argmax(result)
      elif self.answer ==1 :
        print("prediccion: branquias ")
      elif self.answer ==2 :
        print("prediccion: girodactilo ")
      elif self.answer == 3:
        print("prediccion: gusano lernea ")
      elif self.answer ==4 :
        print("prediccion: hidropecia ")
      elif self.answer == 5:
        print("prediccion: hongos")
      elif self.answer ==6 :
        print("prediccion: huecos en la cabesa")
      elif self.answer == 7 :
        print("prediccion: ich ")
      elif self.answer ==8 :
        print("prediccion: no es un pez")
      elif self.answer == 9:
        print("prediccion: ojo picho ")
      elif self.answer == 10:
        print("parasito en la lengua")
      elif self.answer == 11:
        print("prediccion: podredumbre de aletas ")
      elif self.answer == 12:
        print("prediccion: quemadura de bagre ")
      elif self.answer == 13:
        print("prediccion: es un pez sano")
      #pare="noes.jpg"
      #noespez = cv2.imread(pare, cv2.IMREAD_COLOR)
      #noespezres = cv2.resize(noespez,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
      #seleccion = cv2.add(noespezres,imagenpez)
      #cv2.imshow ("ventana2",seleccion)
      return self.answer

        
#curapeces=curapeces()
#curapeces.image()
#curapeces.save_nn()
predict=predict()
print(predict.model)
predict.display_image()
print(predict.predict())
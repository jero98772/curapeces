


from flask import Flask
from flask import render_template
from flask import Flask, render_template, Response
#from tomarfotos import VideoCamera
import numpy as np
import cv2
import time
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
##### importamos librerias nesarias
pez='./especimenesdeprueba/'+input()
#### hacemos que  las imagenes se puedan escojer desde la carpeta especimenesdeprueba

    
imagenpez = cv2.imread(pez, cv2.IMREAD_COLOR)

cv2.imshow ('ventana1',imagenpez)
cv2.waitKey(0)
cv2.destroyAllWindows()
longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)
def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 2:
      print("prediccion: no es un pez")
      #pare="noes.jpg"
      #noespez = cv2.imread(pare, cv2.IMREAD_COLOR)
      #noespezres = cv2.resize(noespez,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
      #seleccion = cv2.add(noespezres,imagenpez)
      #cv2.imshow ('ventana2',seleccion)
  elif answer == 0:
    
       print("prediccion: enfermo")
       if answer == 0:
             print("prediccion:  atcado o tumor y deformidad")
       elif answer == 1:
             print("prediccion: branquias ")
       elif answer == 2:
             print("prediccion: girodactilo ")
       elif answer == 3:
             print("prediccion: gusano lernea ")
       elif answer == 4:
             print("prediccion: hidropecia ")
       elif answer == 5:
             print("prediccion: hongos")
       elif answer == 6:
             print("prediccion: huecos en la cabesa")
       elif answer == 7:
             print("prediccion: ich ")
       elif answer == 8:
             print("prediccion: ojo picho ")
       elif answer == 9:
             print("prediccion: muerto ")
       elif answer == 10:
             print("parasito en la lengua")
       elif answer == 11:
             print("prediccion: podredumbre de aletas ")
       elif answer == 12:
             print("prediccion: quemadura de bagre ")
       elif answer == 13:
             print("prediccion: quemadura de bagre ")


  elif answer== 1:
      print("prediccion: sano")
  

  return answer

predict(pez)

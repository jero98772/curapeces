


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
       # modeloenfemedad = './modeloenfermo/modelopezenfermo.h5'
       # pesos_modeloenfemedad = './modeloenfermo/pesospezenfermo.h5'
       # cnn = load_model(modeloenfemedad)
       # cnn.load_weights(pesos_modeloenfemedad)

modelo_si_es_un_pez = './modelo6/modelo.h5' 

modelopez_sano_o_enefermo ='./modelopez/modelopez.h5'

modelo_de_la_enfemedad ='./modeloenfermo/modelopezenfermo.h5'

pesos_modelo_si_es_un_pez = './modelo6/pesos.h5'

pesospez_sano_o_enefermo ='./modelopez/pesospez.h5'

pesos_modelo_de_la_enfemedad='./modeloenfermo/pesospezenfermo.h5'  

cnn =load_model(modelo_si_es_un_pez)
cnn2 =load_model(modelopez_sano_o_enefermo)
cnn3 =load_model(modelo_de_la_enfemedad)
cnn.load_weights(pesos_modelo_si_es_un_pez) # si es pez 
cnn2.load_weights(pesospez_sano_o_enefermo)# si es cual enfemedad
cnn3.load_weights(pesos_modelo_de_la_enfemedad )#si esta enfermo
def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("prediccion:es un pez")
    x2 = load_img(file, target_size=(longitud, altura))
    x2 = img_to_array(x2)
    x2 = np.expand_dims(x2, axis=0)
    array = cnn2.predict(x2)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
       print("prediccion: es un pez enefmo que tiene")
       x3 = load_img(file, target_size=(longitud, altura))
       x3 = img_to_array(x3)
       x3 = np.expand_dims(x3, axis=0)
       array = cnn3.predict(x3)
       result = array[0]
       answer = np.argmax(result)
       if answer == 0:
           print("prediccion: gusano lernea ")
       elif answer ==1 :
           print("prediccion: hidropecia ")
       elif answer ==2 :
           print("prediccion: huecos en la cabesa")
       elif answer == 3:
           print("prediccion: ich ")
       elif answer ==4 :
           print("prediccion: quemadura de bagre ")
       elif answer == 5:
           print("prediccion:  atcado o tumor y deformidad")
       elif answer ==6 :
           print("prediccion: branquias ")
       elif answer == 7 :
             print("prediccion: girodactilo ")
       elif answer ==8 :
           print("prediccion: hongos")
       elif answer == 9:
           print("prediccion: muerto ")
       elif answer == 10:
           print("prediccion: ojo picho ")
       elif answer == 11:
           print("parasito en la lengua")
       elif answer == 12:
           print("prediccion: podredumbre de aletas ")
    elif answer == 1:
        print("prediccion: es un pez sano")
  elif answer == 1:
    print("prediccion: no es un pez")
      #pare="noes.jpg"
      #noespez = cv2.imread(pare, cv2.IMREAD_COLOR)
      #noespezres = cv2.resize(noespez,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
      #seleccion = cv2.add(noespezres,imagenpez)
      #cv2.imshow ('ventana2',seleccion)


  return answer

predict(pez)

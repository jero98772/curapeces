
import numpy as np
import cv2
import time
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
#import a libraris
#importamos las librerias
pez='./especimenesdeprueba/'+input()
#selcion a imagen in folder who named especimenesdeprueba and the extencion of file and extencion of file ejm .png .jpg
#seleccionamos la imagen que debe estar en la carpeta especimenesdeprueba mas el nombre del archivo y la extecion .png .jpg
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
  if answer == 0:
      print("prediccion: enfermo")
      if answer == 0:
            print("prediccion: girodactilo")
      elif answer == 1:
            print("prediccion: lernaea ")
      elif answer == 2:
            print("prediccion: hidropecia ")
      elif answer == 3:
            print("prediccion: ich ")
  elif answer == 1:
      print("prediccion: no es un pez")
  elif answer== 2:  if answer == 0:
      pezenfermo="enfermo.png"
      enfermo = cv2.imread(pezenfermo, cv2.IMREAD_COLOR)
      seleccion = cv2.addWeighted(imagenpez,enfermo)
      cv2.imshow ('ventana2',seleccion)
      print("prediccion: enfermo")
      if answer == 0:
            print("prediccion: girodactilo")
      elif answer == 1:
            print("prediccion: lernaea ")
      elif answer == 2:
            print("prediccion: hidropecia ")
      elif answer == 3:
            print("prediccion: ich ")
  elif answer == 1:
      print("prediccion: no es un pez")
      pare="noes.jpg"
      noespez = cv2.imread(pare, cv2.IMREAD_COLOR)
      noespezres = cv2.resize(noespez,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
      seleccion = cv2.add(noespezres,imagenpez)
      cv2.imshow ('ventana2',seleccion)
  elif answer== 2:
      print("prediccion: sano")
  return answer

      print("prediccion: sano")
  return answer

predict(pez)

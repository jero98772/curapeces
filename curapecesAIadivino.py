
import numpy as np
import cv2
import time
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

pez='./especimenesdeprueba/'+input()
imgenfemo= "noes.jpg"
imagenpez = cv2.imread(pez, cv2.IMREAD_COLOR)

#imagenpez =  cv2.addWeighted(imagenpez, 0.5, imgenfemo, 0.5, 0)

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
  elif answer== 2:
      print("prediccion: sano")
  return answer

predict(pez)

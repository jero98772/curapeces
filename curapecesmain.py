import argparse
import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2 #importo opencv
pez_pruba = 'ich.jpg' #llamo la variable como la imagen
imagen_pezgris =cv2.imread(pez_pruba, cv2.IMREAD_GRAYSCALE)
imagen_pez =cv2.imread(pez_pruba)
k = 7
sinreflejos =cv2.GaussianBlur(imagen_pez,(k,k),0)
sinreflejosdos =cv2.cvtColor(sinreflejos, cv2.COLOR_BGR2HSV)
blancobajo= np.array([0,140,120] ,)#[0,140,120]
blancoalto = np.array([255,220,255], )#[255,220,255]
mascara = cv2.inRange(sinreflejos,blancobajo,blancoalto)
v1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(15,15))
mascara_tapada = cv2.morphologyEx(mascara,cv2.MORPH_CLOSE,v1)
mascara_retapada = cv2.morphologyEx(mascara_tapada,cv2.MORPH_OPEN,v1)
blancoperfect  = np.array(255)
numerodeblancos = np.sum(mascara_tapada == blancoperfect)
numerodeblancosperfectos = numerodeblancos /100
if numerodeblancosperfectos > 10:
    print ('tiene',numerodeblancosperfectos,' puntos de ick')
else:
    print ('no tine ict')


#proceso la imagn como imagen_pez atraves de la clase
#imread y selecciono la variable

cv2.imshow ('ventana1',imagen_pez)#y decimos que nos muestre imagen_pez y la ventana sellama ventana y +
cv2.imshow ('ventana2',imagen_pezgris)
cv2.imshow ('ventana3',sinreflejos)
cv2.imshow ('ventana4',sinreflejosdos)
cv2.imshow ('ventana5',mascara)
cv2.imshow ('ventana6',mascara_tapada)
cv2.imshow ('ventana7',v1)
cv2.imshow ('ventana8',mascara_retapada)


cv2.waitKey(0)
cv2.destroyAllWindows()

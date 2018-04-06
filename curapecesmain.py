import argparse
import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2 #importo las librerias
k = 7
blancobajo= np.array([0,140,120] ,)#[0,140,120]
blancoalto = np.array([255,220,255], )#[255,220,255]
blancoperfect  = np.array(255)#este esta en escala de grises
#variables
pez_pruba = input() #el lo que pongas en el input va a ser el nombre del archivo
imagen_pezgris =cv2.imread(pez_pruba, cv2.IMREAD_GRAYSCALE)
imagen_pez =cv2.imread(pez_pruba)
sinreflejos =cv2.GaussianBlur(imagen_pez,(k,k),0)
sinreflejosdos =cv2.cvtColor(sinreflejos, cv2.COLOR_BGR2HSV)
mascara = cv2.inRange(sinreflejos,blancobajo,blancoalto)
v1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(15,15))
mascara_tapada = cv2.morphologyEx(mascara,cv2.MORPH_CLOSE,v1)
mascara_retapada = cv2.morphologyEx(mascara_tapada,cv2.MORPH_OPEN,v1)
#las varibles de imagen como son y como van a salir
numerodeblancos1 = np.sum(mascara_tapada == blancoperfect)
numerodeblancosperfectos1 = numerodeblancos1 /100
if numerodeblancosperfectos1 > 10 and numerodeblancosperfectos1 <1000:
    print (' imagen 1 tiene',numerodeblancosperfectos1,' puntos de ick')
else:
    print ('imagen 1 no tine ict')
numerodeblancos2 = np.sum(mascara_retapada == blancoperfect)
numerodeblancosperfectos2 = numerodeblancos2 /100
if numerodeblancosperfectos2 > 10 and numerodeblancosperfectos2 <1000:
    print ('imagen 2 tiene',numerodeblancosperfectos2,' puntos de ick')
else:
    print ('imagen 2 no tine ick')
numerodeblancos3 = np.sum(mascara_tapada == blancoperfect)
numerodeblancosperfectos3 = numerodeblancos3 /100
if numerodeblancosperfectos3 > 10 and numerodeblancosperfectos3 <10000:
    print ('imagen 3 tiene',numerodeblancosperfectos3,' puntos de ick')
else:
    print ('imagen 3 no tine ict')
numerodeblancos4 = np.sum(imagen_pezgris == blancoperfect)
numerodeblancosperfectos4 = numerodeblancos4 /100
if numerodeblancosperfectos4 > 10 and numerodeblancosperfectos4 <1000:
    print ('imagen 4 tiene',numerodeblancosperfectos4,' puntos de ick')
else:
    print ('imagen 4 no tine ick')
#condicionales  para un promedio y solo para algunas varibles de imagens
numerodeblancos = numerodeblancos1 +numerodeblancos2+numerodeblancos3+numerodeblancos4
numerodeblancosperfecto = numerodeblancos /400
if numerodeblancosperfecto > 10 and numerodeblancosperfecto <1000:
    print ('imagenes en promedio tiene',numerodeblancosperfecto,' puntos de ick')
else:
    print (' las  4  imagenes no tine ick pero tine', numerodeblancosperfecto 'puntos de refleojos')
cv2.imshow ('ventana1',imagen_pez)#y decimos que nos muestre imagen_pez y la ventana sellama ventana y +
cv2.imshow ('ventana2',imagen_pezgris)
cv2.imshow ('ventana3',sinreflejos)
cv2.imshow ('ventana4',sinreflejosdos)
cv2.imshow ('ventana5',mascara)
cv2.imshow ('ventana6',mascara_tapada)
cv2.imshow ('ventana7',mascara_retapada)


cv2.waitKey(0)
cv2.destroyAllWindows()

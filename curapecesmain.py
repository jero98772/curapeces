#pastel o ayudas
#cv2.line ( image, starting point , end point , color , line thickness, line type)
#cv2.circle ( image, center, radius, color of border, line thickness / fill type, line type)
#cv2.ellipse ( image, center, axes lengths, rotation degree of ellipse, starting angle , ending angle, color, line thickness / fill type, line type)
#cv2.rectangle ( image, upper left corner vertex, lower right corner vertex, line thickness / fill type, line type)
#cv2.putText ( image, text, starting point of text, font type, font scale, color, linetype )
#PERDONA la orrografia grcias por la compremncion
#importamos las librerias de opencv, numpy ,flask ,etc
#lineas sin entender bien
#173
import cv2
import sys
import argparse
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin
import math
import urllib

from flask import render_template
from flask import Flask
print("camara ;; web ;; imagen ")
#declaramos la varible seccion como un input
seccion = input ()
#si es en la input escribimos web  sale hola mundo  de flask
if seccion =="web":
    print ("inicio ;; todo")
    numero = input()
    if numero == "todo":
        app = Flask(__name__, template_folder = 'teplates')
        @app.route('/')
        def index ():
            return render_template('index.html', name =name)
        if __name__=='__main__':
            app.run(debug = False ,port= 8000)
        pass
    if numero == "inicio":
        app = Flask(__name__)
        @app.route('/')
        def index ():
            return render_template('index.html')
        if __name__=='__main__':
            app.run(debug = False ,port= 8000)

#si escribimos camara vamamos al menu del sugmenu de camara
if seccion == "camara":
#imprimimos usted escogio camara
    print("usted escogio camara")
    print ("menu ;; ich ;; prueba ;;ni idea")
    seleccion = input()
#otro input para seleccionar
    if seleccion== "prueba":
#declaramos la variable captura como video en vivo en colores
        while (True):
            captura = cv2.VideoCapture(0)
            ret,frame = captura.read()
            cv2.imshow("video",frame)
            if(cv2.waitKey(1) & 0xff == ord("q")):
			             break

#PERDONA la orrografia grcias por la comprencion
        captura.relase()
        cv2.destroyAllWindos()
    if seleccion =="ich":
            ap = argparse.ArgumentParser()
            ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
            ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
            args = vars(ap.parse_args())
            lower = {'ich':(0,0,0)}
            upper = {'ich':(255,60,255)}
#PERDONA la orrografia grcias por la compremncion
            colors = {'ich':(0, 6, 100)}


            if not args.get("video", False):
                camera = cv2.VideoCapture(0)



            else:
                camera = cv2.VideoCapture(args["video"])

            while True:

                (grabbed, frame) = camera.read()

                if args.get("video") and not grabbed:
                    break


                blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                for key, value in upper.items():
                    kernel = np.ones((9,9),np.uint8)
                    mask = cv2.inRange(hsv, lower[key], upper[key])
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center = None

                    if len(cnts) > 0:

                        c = max(cnts, key=cv2.contourArea)
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        M = cv2.moments(c)
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


                        if radius > 0.5:

                            cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                            cv2.putText(frame,key + " punto", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)



                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break


            camera.release()
            cv2.destroyAllWindows()

    if seleccion == "ni idea":
        cap = cv2.VideoCapture(0)
        while(1):

#PERDONA la orrografia grcias por la compremncion



                #_, frame = cap.read()
                #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                #lower_white = np.array([0,0,0], dtype=np.uint8)
                #upper_white = np.array([255,37,255], dtype=np.uint8)

                #mask = cv2.inRange(hsv, lower_white, upper_white)
                # Bitwise-AND mask and original image
                #res = cv2.bitwise_and(frame,frame, mask= mask)

            _, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HS)
            lower_white = np.array([0,0,0], dtype=np.uint8)
            upper_white = np.array([255,37,255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_white, upper_white)
            res = cv2.bitwise_and(frame,frame, mask= mask)

            cv2.imshow('frame',frame)
            cv2.imshow('mask',mask)
            cv2.imshow('res',res)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

                cv2.destroyAllWindows()
if seccion == "imagen":
        print ("usted escogio imagen")

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
    print ('imagen 2 no tine ict')
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
    print ('imagen 4 no tine ict')
#condicionales  para un promedio y solo para algunas varibles de imagens
numerodeblancos = numerodeblancos1 +numerodeblancos2+numerodeblancos3+numerodeblancos4
numerodeblancosperfecto = numerodeblancos /400
if numerodeblancosperfecto > 10 and numerodeblancosperfecto <1000:
    print ('imagenes en promedio tiene',numerodeblancosperfecto,' puntos de ick')
else:
    print (' las  4  imagenes no tine ict pero tine', numerodeblancosperfecto, 'puntos de refleojos')
cv2.imshow ('ventana1',imagen_pez)#y decimos que nos muestre imagen_pez y la ventana sellama ventana y +
cv2.imshow ('ventana2',imagen_pezgris)
cv2.imshow ('ventana3',sinreflejos)
cv2.imshow ('ventana4',sinreflejosdos)
cv2.imshow ('ventana5',mascara)
cv2.imshow ('ventana6',mascara_tapada)
cv2.imshow ('ventana7',mascara_retapada)


cv2.waitKey(0)
cv2.destroyAllWindows()

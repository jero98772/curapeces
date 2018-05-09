#pastel o ayudas ;;healps notes
#cv2.line ( image, starting point , end point , color , line thickness, line type)
#cv2.circle ( image, center, radius, color of border, line thickness / fill type, line type)
#cv2.ellipse ( image, center, axes lengths, rotation degree of ellipse, starting angle , ending angle, color, line thickness / fill type, line type)
#cv2.rectangle ( image, upper left corner vertex, lower right corner vertex, line thickness / fill type, line type)
#cv2.putText ( image, text, starting point of text, font type, font scale, color, linetype )
#PERDONA la orrografia grcias por la compremncion ;; and i am dont speek english
import cv2
import sys
import argparse
import math
import urllib
import time
#importamos las librerias nesesarias ;;
import numpy as np
#importamos las librerias nesesarias y las llamamos diferentes
from math import cos, sin
from math import cos, sin
from flask import render_template
from flask import Flask
from datetime import datetime
from flask import Flask, render_template, Response
#importamos de las librerias nesesarias las funciones nesesarias
from matplotlib import pyplot as plt
#importamos de las librerias nesesarias las funciones nesesarias la llamamos diferentes;; import libraris
print("camara ;; web ;; imagen ;;contar")
#imprima las opciones ;;print opstion to write
seccion = input()
app = Flask(__name__)
#declaramos la varible seccion como un input
#si es en la input escribimos web  sale hola mundo  de flask
#si escribimos camara vamamos al menu del sugmenu de camara
if seccion == ("web"):
    print ("inicio ;; todo")
    #imprima las opciones;;print opstion to write

    seleccion = input()
    #declaramos la varible seccion como un input
    if seleccion  == "todo":
        @app.route('/')
        def index():
            return render_template('curarpeces.html')

        def gen():
            i=1
            while i<10:
                yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
                i+=1

    def get_frame():

        camera_port=0

        ramp_frames=100

        camera = cv2.VideoCapture(camera_port) #PERDONA la orrografia grcias por la compremncion ;; and i am dont speek english


        i=1
        while True:
            retval, im = camera.read()
            imgencode=cv2.imencode('.jpg',im)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
            i+=1

        del(camera)

    @app.route('/calc')
    def calc():
        return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


        if __name__ == '__main__':
            app.run(host='localhost', debug=True, threaded=True)


    if seleccion == "inicio":
        @app.route('/')
        def index ():
            return render_template('acercadelproyecto.html')
        if __name__=='__main__':
            app.run(host='localhost', debug=True, threaded=True)


if seccion == "camara":
#imprimimos usted escogio camara
#imprima las opciones
    print("usted escogio camara")
    print (" ich ;; prueba ;;ni idea ;; hidropecia" )
    seleccion = input()
#otro input para seleccionar
    if seleccion== "prueba":
#declaramos la variable captura como video en vivo en colores
        cap = cv2.VideoCapture(0)
        while (True):
            _, frame = cap.read()
            cv2.imshow("video",frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:

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


if seccion == "imagen":
    print ("ich ;; hidropecia")
seleccion = input()
if seleccion == "ich":
    print ("select imagen ;; selecciona la imagen")
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
    cv2.imshow ('ventana1',imagen_pez)#y decimos que nos muestre imagen_pez y la ventana sellama ventana
    cv2.imshow ('ventana2',imagen_pezgris)
    cv2.imshow ('ventana3',sinreflejos)
    cv2.imshow ('ventana4',sinreflejosdos)
    cv2.imshow ('ventana5',mascara)
    cv2.imshow ('ventana6',mascara_tapada)
    cv2.imshow ('ventana7',mascara_retapada)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
if seleccion == "hidropecia":
    pez_pruba = input()

    hidropecia = cv2.imread(pez_pruba, cv2.IMREAD_GRAYSCALE)
    dectectordehidropecia = cv2.SimpleBlobDetector()
    punosclaves = dectectordehidropecia.detect(hidropecia)
    dibujadehidropecia = cv2.drawKeypoints(hydropecia, punosclaves, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("ventana 1", dectectordehidropecia)
    cv2.waitKey(0)

if seccion == "contar":
    seleccion = input()
    if seleccion == "video":
        if __name__ == '__main__':

                nada = "nada"
    # Are we finding motion or tracking
    status = 'motion'
    # How long have we been tracking
    idle_time = 0

    # Background for motion detection
    back = None
    # An MIL tracker for when we find motion
    tracker = cv2.TrackerMIL_create()

    # Webcam footage (or video)
    video = cv2.VideoCapture("contar.mp4")

    # LOOP
    while True:
        # Check first frame
        ok, frame = video.read()

        # Grayscale footage
        grayi = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Blur footage to prevent artifacts
        grayii = cv2.GaussianBlur(gray,(21,21),0)
        blancobajo= np.array([0,140,120] ,)#[0,140,120]
        blancoalto = np.array([255,220,255], )#[255,220,255]
        blancoperfectA  = np.array(255)#este esta en escala de grises
        blancoperfectB  = np.array(205)#este esta en escala de grises
        v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(15,15))
        mascara = cv2.inRange(grayii,blancoperfectA,blancoperfectB)
        mascara_tapada = cv2.morphologyEx(mascara,cv2.MORPH_CLOSE,v)
        numerodeblancos1 = np.sum(mascara_tapada)
        if numerodeblancos1 < 0:
            print (' imagen 1 tiene',numerodeblancos1,'peces')
        else:
            print ('no hay peces con el color')
        # Check for background
        if back is None:
            # Set background to current frame
            back = gray

        if status == 'motion':
            # Difference between current frame and background
            frame_delta = cv2.absdiff(back,gray)
            # Create a threshold to exclude minute movements
            thresh = cv2.threshold(frame_delta,25,255,cv2.THRESH_BINARY)[1]

            #Dialate threshold to further reduce error
            thresh = cv2.dilate(thresh,None,iterations=2)
            # Check for contours in our threshold
            _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


            # Check each contour
            if len(cnts) != 0:
                # If the contour is big enough

                # Set largest contour to first contour
                largest = 0

                # For each contour
                for i in range(len(cnts)):
                    # If this contour is larger than the largest
                    if i != 0 & int(cv2.contourArea(cnts[i])) > int(cv2.contourArea(cnts[largest])):
                        # This contour is the largest
                        largest = i

                if cv2.contourArea(cnts[largest]) > 1000:
                    # Create a bounding box for our contour
                    (x,y,w,h) = cv2.boundingRect(cnts[0])
                    # Convert from float to int, and scale up our boudning box
                    (x,y,w,h) = (int(x),int(y),int(w),int(h))
                    # Initialize tracker
                    bbox = (x,y,w,h)
                    ok = tracker.init(frame, bbox)
                    # Switch from finding motion to tracking
                    status = 'tracking'


        # If we are tracking
        if status == 'tracking':
            # Update our tracker
            ok, bbox = tracker.update(frame)
            # Create a visible rectangle for our viewing pleasure
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame,p1,p2,(0,0,255),10)


        # Show our webcam
        cv2.imshow("Camera",frame)


        # If we have been tracking for more than a few seconds
        if idle_time >= 30:
            # Reset to motion
            status = 'motion'
            # Reset timer
            idle_time = 0

            # Reset background, frame, and tracker
            back = None
            tracker = None
            ok = None

            # Recreate tracker
            tracker = cv2.TrackerMIL_create()


        # Incriment timer
        idle_time += 1


# Check if we've quit
    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty('Camera',0) == -1:

#QUIT
        video.release()
    cv2.destroyAllWindows()
    seleccion = input()
    if seleccion == "camara":
        if __name__ == '__main__':
    # Are we finding motion or tracking
            status = 'motion'
    # How long have we been tracking
    idle_time = 0

    # Background for motion detection
    back = None
    # An MIL tracker for when we find motion
    tracker = cv2.TrackerMIL_create()

    # Webcam footage (or video)
    video = cv2.VideoCapture(0)

    # LOOP
    while True:
        # Check first frame
        ok, frame = video.read()

        # Grayscale footage
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Blur footage to prevent artifacts
        gray = cv2.GaussianBlur(gray,(21,21),0)

        # Check for background
        if back is None:
            # Set background to current frame
            back = gray

        if status == 'motion':
            # Difference between current frame and background
            frame_delta = cv2.absdiff(back,gray)
            # Create a threshold to exclude minute movements
            thresh = cv2.threshold(frame_delta,25,255,cv2.THRESH_BINARY)[1]

            #Dialate threshold to further reduce error
            thresh = cv2.dilate(thresh,None,iterations=2)
            # Check for contours in our threshold
            _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


            # Check each contour
            if len(cnts) != 0:
                # If the contour is big enough

                # Set largest contour to first contour
                largest = 0

                # For each contour
                for i in range(len(cnts)):
                    # If this contour is larger than the largest
                    if i != 0 & int(cv2.contourArea(cnts[i])) > int(cv2.contourArea(cnts[largest])):
                        # This contour is the largest
                        largest = i

                if cv2.contourArea(cnts[largest]) > 1000:
                    # Create a bounding box for our contour
                    (x,y,w,h) = cv2.boundingRect(cnts[0])
                    # Convert from float to int, and scale up our boudning box
                    (x,y,w,h) = (int(x),int(y),int(w),int(h))
                    # Initialize tracker
                    bbox = (x,y,w,h)
                    ok = tracker.init(frame, bbox)
                    # Switch from finding motion to tracking
                    status = 'tracking'


        # If we are tracking
        if status == 'tracking':
            # Update our tracker
            ok, bbox = tracker.update(frame)
            ok, bbox = tracker.update(mascara_tapada)
            # Create a visible rectangle for our viewing pleasure
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame,p1,p2,(0,0,255),10)


        # Show our webcam
        cv2.imshow("Camera",frame)
        cv2.imshow("Camera",mascara_tapada)

        # If we have been tracking for more than a few seconds
        if idle_time >= 30:
            # Reset to motion
            status = 'motion'
            # Reset timer
            idle_time = 0

            # Reset background, frame, and tracker
            back = None
            tracker = None
            ok = None

            # Recreate tracker
            tracker = cv2.TrackerMIL_create()


        # Incriment timer
        idle_time += 1

    blancobajo= np.array([0,140,120] ,)#[0,140,120]
    blancoalto = np.array([255,220,255], )#[255,220,255]
    blancoperfect  = np.array(255)#este esta en escala de grises
    mascara = cv2.inRange(gray,blancobajo,blancoalto)
    mascara_tapada = cv2.morphologyEx(mascara,cv2.MORPH_CLOSE,v1)
    blancoperfect  = np.array(255)
    numerodeblancos1 = np.sum(mascara_tapada == blancoperfect)
    if numerodeblancos1 > 0:
        print (' imagen 1 tiene',numerodeblancos1,'peces')
    else:
        print ('no hay peces con el color')
        # Check if we've quit
        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty('Camera',0) == -1:
#QUIT
            video.release()
    cv2.destroyAllWindows()

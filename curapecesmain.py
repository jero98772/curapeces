import cv2
import sys
import argparse
import numpy as np
import math
import urllib
seccion = raw_input ()
if seccion == "camara":
    print ("usted escogio camara")
    seleccion = raw_input()
    if seleccion== "prueba":
        captura = cv2.VideoCapture(0)
        while (True):
		ret,frame = captura.read()
		cv2.imshow("video",frame)
		if(cv2.waitKey(1) & 0xff == ord("q")):
			             break
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
            upper = {'ich':(0,0,255)}


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
                            print "este pez tiene ich"


                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break


            camera.release()
            cv2.destroyAllWindows()

    if seleccion == "ni idea":
        cap = cv2.VideoCapture(0)
        while(1):

            _, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HS
            lower_white = np.array([0,0,0], dtype=np.uint8)
            upper_white = np.array([255,37,255], dtype=np.uint
            mask = cv2.inRange(hsv, lower_white, upper_whit
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

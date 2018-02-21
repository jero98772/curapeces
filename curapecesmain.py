#pastel o ayudas
#cv2.line ( image, starting point , end point , color , line thickness, line type)
#cv2.circle ( image, center, radius, color of border, line thickness / fill type, line type)
#cv2.ellipse ( image, center, axes lengths, rotation degree of ellipse, starting angle , ending angle, color, line thickness / fill type, line type)
#cv2.rectangle ( image, upper left corner vertex, lower right corner vertex, line thickness / fill type, line type)
#cv2.putText ( image, text, starting point of text, font type, font scale, color, linetype )

from flask import Flask
import cv2
import sys
import argparse
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin
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
            upper = {'ich':(255,60,255)}

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





                #_, frame = cap.read()
                #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # define range of white color in HSV
                # change it according to your need !
                #lower_white = np.array([0,0,0], dtype=np.uint8)
                #upper_white = np.array([255,37,255], dtype=np.uint8)

                # Threshold the HSV image to get only white colors
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
        green = (0, 255, 0)
            #este esun codigo de siraj raval con el cual estoy aprendiendo  gracias por su comprecion

        def show(image):
            # Figure size in inches
            plt.figure(figsize=(10, 10))

            # Show image, with nearest neighbour interpolation
            plt.imshow(image, interpolation='nearest')

        def overlay_mask(mask, image):
        	#make the mask rgb
            rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            #calculates the weightes sum of two arrays. in our case image arrays
            #input, how much to weight each.
            #optional depth value set to 0 no need
            img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
            return img

        def find_biggest_contour(image):
            # Copy
            image = image.copy()
            #input, gives all the contours, contour approximation compresses horizontal,
            #vertical, and diagonal segments and leaves only their end points. For example,
            #an up-right rectangular contour is encoded with 4 points.
            #Optional output vector, containing information about the image topology.
            #It has as many elements as the number of contours.
            #we dont need it
            contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Isolate largest contour
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

            mask = np.zeros(image.shape, np.uint8)
            cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
            return biggest_contour, mask

        def circle_contour(image, contour):
            # Bounding ellipse
            image_with_ellipse = image.copy()
            #easy function
            ellipse = cv2.fitEllipse(contour)
            #add it
            cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.CV_AA)
            return image_with_ellipse

        def find_strawberry(image):
            #RGB stands for Red Green Blue. Most often, an RGB color is stored
            #in a structure or unsigned integer with Blue occupying the least
            #second least, and Red the third least. BGR is the same, except the
            #order of areas is reversed. Red occupies the least significant area,
            # Green the second (still), and Blue the third.
            # we'll be manipulating pixels directly
            #most compatible for the transofrmations we're about to do
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Make a consistent size
            #get largest dimension
            max_dimension = max(image.shape)
            #The maximum window size is 700 by 660 pixels. make it fit in that
            scale = 700/max_dimension
            #resize it. same width and hieght none since output is 'image'.
            image = cv2.resize(image, None, fx=scale, fy=scale)

            #we want to eliminate noise from our image. clean. smooth colors without
            #dots
            # Blurs an image using a Gaussian filter. input, kernel size, how much to filter, empty)
            image_blur = cv2.GaussianBlur(image, (7, 7), 0)
            #t unlike RGB, HSV separates luma, or the image intensity, from
            # chroma or the color information.
            #just want to focus on color, segmentation
            image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

            # Filter by colour
            # 0-10 hue
            #minimum red amount, max red amount
            min_red = np.array([0, 100, 80])
            max_red = np.array([10, 256, 256])
            #layer
            mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

            #birghtness of a color is hue
            # 170-180 hue
            min_red2 = np.array([170, 100, 80])
            max_red2 = np.array([180, 256, 256])
            mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

            #looking for what is in both ranges
            # Combine masks
            mask = mask1 + mask2

            # Clean up
            #we want to circle our strawberry so we'll circle it with an ellipse
            #with a shape of 15x15
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            #morph the image. closing operation Dilation followed by Erosion.
            #It is useful in closing small holes inside the foreground objects,
            #or small black points on the object.
            mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            #erosion followed by dilation. It is useful in removing noise
            mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

            # Find biggest strawberry
            #get back list of segmented strawberries and an outline for the biggest one
            big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)

            # Overlay cleaned mask on image
            # overlay mask on image, strawberry now segmented
            overlay = overlay_mask(mask_clean, image)

            # Circle biggest strawberry
            #circle the biggest one
            circled = circle_contour(overlay, big_strawberry_contour)
            show(circled)

            #we're done, convert back to original color scheme
            bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

            return bgr

        #read the image
        image = cv2.imread('berry.jpg')
        #detect it
        result = find_strawberry(image)
        #write the new image
        cv2.imwrite('berry2.jpg', result)

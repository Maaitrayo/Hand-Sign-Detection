import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from regex import F


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5" , "Model/labels.txt")

offset = 20
imgsize = 300

folder = "Data/C"
counter = 0

labels = ["A" , "B" , "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgwhite = np.ones((imgsize, imgsize,3), np.uint8)*255
        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgcropShape = imgcrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgsize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgcrop, (wCal, imgsize))
            imgResizeShape = imgResize.shape
            wgap = math.ceil((300-wCal)/2)
            imgwhite[:, wgap:wCal+wgap] = imgResize
            
            prediction, index =  classifier.getPrediction(imgwhite, draw=False)
            print(prediction, index)

        else:
            k = imgsize/w  
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgcrop, (imgsize, hCal))
            imgResizeShape = imgResize.shape
            hgap = math.ceil((300-hCal)/2)
            imgwhite[hgap:hCal+hgap,:] = imgResize

            prediction, index =  classifier.getPrediction(imgwhite, draw=F)
              
        cv2.putText(imgOutput, labels[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX,2, (255,0,255), 2)
        cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset,y+h+offset ), (255,0,255),4)



        cv2.imshow("Image Crop", imgcrop)
        cv2.imshow("Image white", imgwhite)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

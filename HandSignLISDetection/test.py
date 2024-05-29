import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 500

folder = "Data/C"
counter = 0

labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:                                                                   #bounding box
        hand = hands[0]
        x, y, w, h = hand['bbox']  

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset: y + h + offset,                               #w = width h = height
                      x - offset : x + w + offset]                              #matrix
        
        imgCropShape = imgCrop.shape

        #imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop


        aspectRatio = h / w  #height is greater 

        if aspectRatio > 1 :

            k = imgSize / h
            wCal = math.ceil(k * w) #prendo l'intero successivo
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal ) / 2) #gap to put foreward to center the image
            #imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
            imgWhite[ : , wGap : wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)

        else:

            k = imgSize / w
            hCal = math.ceil(k * h) #prendo l'intero successivo
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal ) / 2) #gap to put foreward to center the image
            imgWhite[hGap : hGap + hCal , : ] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw = False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset), (255,0,255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x + 10, y - 25), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255,0,255), 4)
        

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)










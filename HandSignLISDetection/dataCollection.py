import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 20
imgSize = 500

folder = "Data/C"
counter = 0

while True:
    success, img = cap.read()
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

        else:

            k = imgSize / w
            hCal = math.ceil(k * h) #prendo l'intero successivo
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal ) / 2) #gap to put foreward to center the image
            imgWhite[hGap : hGap + hCal , : ] = imgResize




        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)















import cv2
from random import randrange


face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('avengers.jpg')

#blc_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

coord = face_data.detectMultiScale(img)

for [x, y, w, h,] in coord:
    cv2.rectangle(img, (x, y), (x+w , y+h), (randrange(255), randrange(255), randrange(255)), 4)

print(coord)

cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()



    




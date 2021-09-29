import cv2
from random import randrange


face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



vid = cv2.VideoCapture(0)



while True:
    frame_read, frame = vid.read(0) 

    blc_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    coord = face_data.detectMultiScale(frame)   

    print(coord)
    
    for [x, y, w, h,] in coord:
        cv2.rectangle(frame, (x, y), (x+w , y+h), (randrange(255), randrange(255), randrange(255)), 4)
    
    cv2.imshow('Clever Programmer Face Detector', frame)
    k=cv2.waitKey(1)

    if k==ord('q'):
     break

vid.release()
cv2.destroyAllWindows()
    

    
    


    




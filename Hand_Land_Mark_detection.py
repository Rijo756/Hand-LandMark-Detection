import cv2
import time
import mediapipe as mp


#Defining Webcam as video input
cap = cv2.VideoCapture(1)

#defining hand module
mp_Hand = mp.solutions.hands
hands = mp_Hand.Hands() #static_image_mode=False, max_num_hands=2, min_detection_confidence = 0.5,min_tracking_confidence=0.5
mp_draw = mp.solutions.drawing_utils

ptime = 0
ctime = 0

while True:
    #reading from webcam
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #getting the hands from the image using media
    results = hands.process(imgRGB)

    #uding mediapipe module to draw the landmarks into image
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:   
            mp_draw.draw_landmarks(img,handlms, mp_Hand.HAND_CONNECTIONS)



    #to calculate fps
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,"FPS: "+str(round(fps)),(10,20),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
    cv2.imshow("CAM001",img)
    cv2.waitKey(1)

#imports
import cv2
import mediapipe as mp
import time
import math
import osascript as osascript


#functions
def calculateDistance(x1,y1,x2,y2):
    '''this function works out the distance between two coordinates. Im using this to calculate the distance between my thumb and my index finger'''
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def midpoint(x1,y1,x2,y2):
    '''This function returns the midpoint of two coordinates. This helps me find the midpoint to place the text on the line for visual benefit'''
    midx = (x1+x2)/2
    midy = (y1+y2)/2
    return midx, midy


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

distance = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                tipx = int(handLms.landmark[8].x * w)
                tipy = int(handLms.landmark[8].y * h)
                cx, cy = int(lm.x *w), int(lm.y*h)
                if id == 4:

                    distance = calculateDistance(cx, cy, tipx, tipy)
                    mx, my = midpoint(cx, cy, tipx, tipy)
                    cv2.circle(img, (cx,cy), 10, (224,186,255), cv2.FILLED)
                    cv2.line(img, (cx, cy), (tipx,tipy), (0,255,0), 10)


                    #These if statments help turn the disatnce into a percentage to change volume 
                    if distance < 50:
                        volume = 0
                    elif distance > 300:
                        volume = 100
                    else:
                        volume = int((distance/350)*100)

                    dist_vol = "{} / {}%".format(str(int(distance)), volume)

                    cv2.putText(img, dist_vol, (int(mx), int(my)), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),3)

                    vol = "set volume output volume {}".format(volume)
                    osascript.osascript(vol)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cv2.imshow("Image", img)

    cv2.waitKey(1)

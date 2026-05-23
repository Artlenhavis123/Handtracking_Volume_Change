import cv2
import math
import os
import time
import urllib.request

import mediapipe as mp
import osascript
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def calculateDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def midpoint(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_volume = -1
start = time.time()

while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    timestamp_ms = int((time.time() - start) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        h, w, _ = img.shape
        for hand in result.hand_landmarks:
            thumb = hand[4]
            index = hand[8]
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            index_x, index_y = int(index.x * w), int(index.y * h)

            distance = calculateDistance(thumb_x, thumb_y, index_x, index_y)
            mx, my = midpoint(thumb_x, thumb_y, index_x, index_y)

            cv2.circle(img, (thumb_x, thumb_y), 10, (224, 186, 255), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 10)

            if distance < 50:
                volume = 0
            elif distance > 300:
                volume = 100
            else:
                volume = int((distance / 350) * 100)

            dist_vol = "{} / {}%".format(int(distance), volume)
            cv2.putText(img, dist_vol, (int(mx), int(my)),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            if volume != last_volume:
                osascript.osascript("set volume output volume {}".format(volume))
                last_volume = volume

            # Skeleton overlay (manual draw — keeps deps minimal)
            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                a, b = hand[conn.start], hand[conn.end]
                cv2.line(img,
                         (int(a.x * w), int(a.y * h)),
                         (int(b.x * w), int(b.y * h)),
                         (224, 224, 224), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

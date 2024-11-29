import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pygame

def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)

def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5

# Don't open this function)
def secret():
    pygame.mixer.init()
    image = cv2.imread('sys error.jpg')
    cv2.namedWindow('error', cv2.WINDOW_NORMAL)

    cv2.waitKey(1)

    cv2.setWindowProperty('error', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    pygame.mixer.music.load('err1.mp3')
    pygame.mixer.music.play()
    pygame.mixer.music.load('err2.mp3')
    pygame.mixer.music.play()
    cv2.imshow('error', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def V(landmarks):
    if landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        return True
    return False

handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
count = 0
prev_fist = False

screen_width, screen_height = pyautogui.size()

while (cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    results = handsDetector.process(flippedRGB)

    if results.multi_hand_landmarks is not None:
        if len(results.multi_handedness) == 1:
            landmarks = results.multi_hand_landmarks[0].landmark
            landmarks2 = []
        else:
            landmarks = results.multi_hand_landmarks[0].landmark
            landmarks2 = results.multi_hand_landmarks[1].landmark
        print("1:", len(landmarks))
        print("2:", len(landmarks2))
        points = get_points(landmarks, flippedRGB.shape)

        cv2.drawContours(flippedRGB, [points], 0, (255, 0, 0), 2)
        (x, y), r = cv2.minEnclosingCircle(points)
        ws = palm_size(landmarks, flippedRGB.shape)
        if V(landmarks):
            secret()
        if 2 * r / ws > 1.4:
            cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 0, 255), 2)
            prev_fist = False
        else:
            cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 255, 0), 2)
            if not prev_fist:
                count += 1
                print(f"Click {count}")
                if len(landmarks2) != 0:
                    pyautogui.click()
                pyautogui.click()
                prev_fist = True
        cv2.putText(flippedRGB, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

        xp = int(landmarks[0].x * flippedRGB.shape[1])
        yp = int(landmarks[0].y * flippedRGB.shape[0])
        pyautogui.moveTo(xp * screen_width / flippedRGB.shape[1], yp * screen_height / flippedRGB.shape[0])

    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

handsDetector.close()
cap.release()
cv2.destroyAllWindows()

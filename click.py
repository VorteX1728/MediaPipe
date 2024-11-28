import cv2
import mediapipe as mp
import numpy as np
import pyautogui


# Функция для получения точек для рисования контура
def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)


# Функция для вычисления размера ладони
def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5


# Создаем детектор
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
count = 0
prev_fist = False

# Получаем размер экрана для масштабирования координат
screen_width, screen_height = pyautogui.size()

while (cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

    flipped = np.fliplr(frame)
    # Переводим в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    # Распознаем
    results = handsDetector.process(flippedRGB)

    # Рисуем распознанное, если распозналось
    if results.multi_hand_landmarks is not None:
        landmarks = results.multi_hand_landmarks[0].landmark
        points = get_points(landmarks, flippedRGB.shape)

        cv2.drawContours(flippedRGB, [points], 0, (255, 0, 0), 2)
        (x, y), r = cv2.minEnclosingCircle(points)
        ws = palm_size(landmarks, flippedRGB.shape)

        if 2 * r / ws > 1.3:
            cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 0, 255), 2)  # кулак разжат
            prev_fist = False
        else:
            cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 255, 0), 2)  # кулак сжат
            if not prev_fist:
                # Если кулак сжался, увеличиваем счётчик кликов
                count += 1
                print(f"Click {count}")
                pyautogui.click()  # Выполняем клик мышью
                prev_fist = True

        # Рисуем счётчик на экране
        cv2.putText(flippedRGB, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

        xp = int(landmarks[0].x * flippedRGB.shape[1])
        yp = int(landmarks[0].y * flippedRGB.shape[0])
        pyautogui.moveTo(xp * screen_width / flippedRGB.shape[1], yp * screen_height / flippedRGB.shape[0])

    # Переводим обратно в BGR и показываем результат
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

# Освобождаем ресурсы
handsDetector.close()
cap.release()
cv2.destroyAllWindows()
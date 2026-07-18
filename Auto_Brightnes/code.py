import time
import numpy as np
import screen_brightness_control as sbc
import time
import cv2

camera = cv2.VideoCapture(0)
brightness_history = []

try:
    while True:

        # Skip a few frames
        for _ in range(5):
            success, image = camera.read()

        if not success:
            print("Camera Error")
            time.sleep(10)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b = np.mean(gray)

        brightness_history.append(b)
        if len(brightness_history) > 5:
            brightness_history.pop(0)

        avg_brightness = np.mean(brightness_history)

        target = int((avg_brightness / 255) * 100)
        target = max(15, min(100, target))

        current = sbc.get_brightness()[0]

        if abs(current - target) >= 10:
            sbc.set_brightness(target)

        print(
            f"Ambient: {avg_brightness:.1f} | "
            f"Current: {current}% | "
            f"Target: {target}%"
        )

        time.sleep(300)   # 5 minutes

except KeyboardInterrupt:
    print("Stopped.")

finally:
    camera.release()
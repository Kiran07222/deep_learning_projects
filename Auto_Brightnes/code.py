import time
import numpy as np
import screen_brightness_control as sbc
import time
import cv2

camera = cv2.VideoCapture(0)
brightness_history = []

try:
    while True:

    
        success, image = camera.read()
        camera.release()

        if  success:
            print("Camera works")
        else:
            print("camera not workds")
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b = np.mean(gray)

        brightness_history.append(b)
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

except KeyboardInterrupt:
    print("Stopped.")

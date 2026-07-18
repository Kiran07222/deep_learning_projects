import time
import numpy as np
import screen_brightness_control as sbc
import time
import cv2

while 1:
    camera=cv2.VideoCapture(0)
    success,image=camera.read()
    camera.release()

    if success:

        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        b=np.mean(gray)
        current = sbc.get_brightness()[0]
        target = int((b / 255) * 100)
        target = max(15, min(100, target))
        if abs(current - target) >= 10:
            sbc.set_brightness(target)
    else:
        print("unable to capture the image rigth now")
    exit()
    time.sleep(500)
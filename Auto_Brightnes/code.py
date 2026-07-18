import time
import threading
from collections import deque
import numpy as np
import screen_brightness_control as sbc
import cv2


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Unable to open camera")
        return

    # Request a small capture resolution — cuts read + decode time a lot
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    brightness_history = deque(maxlen=30)

    # Track brightness locally instead of querying the OS every loop
    try:
        current = int(sbc.get_brightness()[0])
    except Exception as e:
        print("Could not read current brightness:", e)
        current = 50  # reasonable fallback

    lock = threading.Lock()
    setting_in_progress = False

    def apply_brightness(value):
        nonlocal current, setting_in_progress
        try:
            sbc.set_brightness(value)
            with lock:
                current = value
        except Exception as e:
            print("Failed to set brightness:", e)
        finally:
            with lock:
                setting_in_progress = False

    try:
        while True:
            success, image = camera.read()
            if not success or image is None:
                print("Failed to read from camera")
                break

            # Downscale further before computing mean — 320x240 -> 80x60 is plenty
            small = cv2.resize(image, (80, 60), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            b = float(np.mean(gray))

            brightness_history.append(b)
            avg_brightness = float(np.mean(brightness_history))

            target = int((avg_brightness / 255) * 100)
            target = max(15, min(100, target))

            with lock:
                busy = setting_in_progress
                cur = current

            if not busy and abs(cur - target) >= 10:
                with lock:
                    setting_in_progress = True
                # Run the slow OS call off the main loop so it doesn't block frame capture
                threading.Thread(target=apply_brightness, args=(target,), daemon=True).start()

            print(
                f"Ambient: {avg_brightness:.1f} | "
                f"Current: {cur}% | "
                f"Target: {target}%"
            )

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
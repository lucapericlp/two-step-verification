from typing import Callable
import cv2
from PIL import Image
import numpy as np

def stream(inference: Callable[[np.ndarray], np.ndarray]) -> None:
    cap = cv2.VideoCapture(0)

    try:
        while True:
            print("Capturing frame...")
            ret, _frame = cap.read()
            #  frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Frame', _frame)
            frame_prime = inference(_frame)

            if not ret:
                print("Failed to capture frame")
                break

            cv2.imshow('Frame', frame_prime)
            # Break the loop if 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the VideoCapture object and close the windows.
        cap.release()
        cv2.destroyAllWindows()

import os
import time

from ultralytics import YOLO
import cv2
import numpy as np

video_dir = os.path.join('.', 'videos')

video_path = os.path.join(video_dir, '') #Put the name of video in the empty string
video_output = '{}_out.mp4'.format(video_path)

capture = cv2.VideoCapture(video_path)
ret, frame = capture.read()
Height, Width, _ = frame.shape
output = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'MP4V'), int(capture.get(cv2.CAP_PROP_FPS)), (Width, Height))

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    output.write(frame)
    ret, frame = capture.read()

capture.release()
output.release()
cv2.destroyAllWindows()

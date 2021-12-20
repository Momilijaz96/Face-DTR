from face_detector import YoloDetector
import torch.nn as nn
import torch
import cv2
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
size = 480  # size = 320 or 480

# image standardization
def standardize(image):
    image = (image - 127.5) / 128.0
    return image

# detection model
model = YoloDetector(target_size=size, gpu=0, min_face=20)   # =>set gpu=0 for GPU =>set gpu=-1 for CPU


video_capture = cv2.VideoCapture('/dev/video0')

while True:
    ret, frame = video_capture.read()
    w, h, c = frame.shape
    # resize frame
    image = cv2.resize(frame, (size, size))

    # detection
    start = time.time()
    boxes, _ = model.predict(image)

    end = time.time()
    fps = 'FPS : ' + str(int(1/(end - start)))

    if len(boxes) != 0:
        i = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * h/size)
            y1 = int(y1 * w/size)
            x2 = int(x2 * h/size)
            y2 = int(y2 * w/size)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            i += 1

    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

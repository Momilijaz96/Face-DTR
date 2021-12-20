from face_detector import YoloDetector
import torch.nn as nn
import torch
import cv2
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
size = 480  # size = 320 or 480

# read embeddings
sai = torch.load(f'weights/embeddings/sai_{size}.pt')
umar = torch.load(f'weights/embeddings/umar_{size}.pt')
momal = torch.load(f'weights/embeddings/momal_{size}.pt')
class_data = [sai, umar, momal]
cd_len = len(class_data)
labels = ['sai', 'umar', 'momal']

# similarity
cos = nn.CosineSimilarity()
euc = nn.PairwiseDistance(p=2)

def euclidean(x, embeds):
    out = 2
    dist = []
    for i in embeds:
        d = euc(x, i).item()
        d = round(d, 2)
        dist.append(d)
    m = min(dist)
    m1 = max(dist)
    if m < 1 and m1 < 1.5:
        out = m
    return out

# image standardization
def standardize(image):
    image = (image - 127.5) / 128.0
    return image

# detection model
model = YoloDetector(target_size=size, gpu=0, min_face=20)   # =>set gpu=0 for GPU =>set gpu=-1 for CPU

# recognition model
resnet = torch.load(f'weights/resnet_{size}.pt', map_location=torch.device(device))
resnet = resnet.eval().to(device)
resnet.classify = False

video_capture = cv2.VideoCapture('/dev/video0')

while True:
    ret, frame = video_capture.read()
    w, h, c = frame.shape
    # resize frame
    image = cv2.resize(frame, (size, size))

    # detection
    start = time.time()
    boxes, _ = model.predict(image)

    # recognition pre-processing
    if len(boxes) != 0:
        aligned = []
        for box in boxes:
            img = image[box[1]:box[3], box[0]:box[2], :]
            img = cv2.resize(img, (160, 160))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).to(device)
            img = standardize(img)
            aligned.append(img)
        data = torch.stack(aligned).to(device)

        # recognition
        embeddings = resnet(data).detach().cpu()
 
    # post-processing recognition output
    predictions = []
    if len(boxes) != 0:
        for i in range(len(embeddings)):
            x = embeddings[i].unsqueeze(0)

            # cosine similarity
            '''
            maxx = []
            for j in range(cd_len):
                cs = cos(x, class_data[j])
                temp = cs.max().item()
                if temp > 0.45:
                    maxx.append(temp)
                else:
                    maxx.append(-1)
            maxx_m = max(maxx)
            if maxx_m != -1:
                k = maxx.index(maxx_m)
                predictions.append(labels[k])
            else:
                predictions.append('unknown')
            '''

            # euclidean
            minn = []
            for j in range(cd_len):
                e = euclidean(x, class_data[j])
                minn.append(e)
            minn_m = min(minn)
            if minn_m != 2:
                k = minn.index(minn_m)
                predictions.append(labels[k])
            else:
                predictions.append('unknown')


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
            cv2.putText(frame, predictions[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
            i += 1

    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

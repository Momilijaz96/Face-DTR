# Face-DTR
This repo covers implementation of face detection, recognition and tracking  on Jetson Xavier, with above 15 FPS. 
* Face detection is performed by [Yolo5Face](https://github.com/elyha7/yoloface).
* Face recognition is performed Inception ResNetv3, recognition module from [facenet-pytorch](https://github.com/timesler/facenet-pytorch#use-this-repo-in-your-own-git-project).
* Tracking is performed by DeepSort Tracker from [deep_sort pytorch](https://github.com/ZQPei/deep_sort_pytorch).
<p align='center'>
<img src="images/res2.png" width="23%"></img>
<img src="images/res3.png" width="23%"></img>
<img src="images/res4.png" width="23%"></img>
<img src="images/res5.png" width="23%"></img> 
</p>

<h2>Detection</h2>
<p>
Detection is performed by a wrap over of original github repo for [Yolo5Face: Why reinventing a face detector?] paper(https://github.com/deepcam-cn/yolov5-face). This paper implements a yolov5n with wing loss, which is trained over WIDEFace dataset, to perform face detection.
Face detection with this repo was pretty smooth and robust to illumination changes, side poses and varying resolution of faces. 
</p>

<h2>Recogntion</h2>
<p>
  Recognition is performed using InceptionResNetv3, which is pretrained on VGGFace2, and we finetuned on faces of the people we wanted to recognize, in this case, we were three team members, whose faces we wanted algo to recognize, so we took 40 different images of each person, with varying backgrounds, clothes, illumination and face poses and sizes. In total, this model was fine tuned on 130 images roughly for 30 epochs.  
</p>
<p>
  Recogniton using above mentioend detector is performed as follows:
  
  * After fine tuning model, 3 different images of each person were passed though the model, one face from front, and other from left anf right side poses, and output of the final layer of trained Inception ResNet are stored in a file.
  * At time of recognition, the detected faces from yolov5 detector are cropped and saved.
  * These cropped faces are passed through this inception resnet to get face embeddings.
  * These embeddings are then compared with the saved embeddings using euclidean distannce. If embeddings of the faces in the frame does not cross similarity threshold person is declared as unnown otherwise, closest embedding are used to assign label.

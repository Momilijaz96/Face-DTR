# Face-DTR
This repo performs face detection, recognition and tracking  on Jetson Xavier, with above 15 FPS. 
* Face detection is performed by [Yolo5Face](https://github.com/elyha7/yoloface).
* Face recognition is performed Inception ResNetv3, recognition module from [facenet-pytorch](https://github.com/timesler/facenet-pytorch#use-this-repo-in-your-own-git-project).
* Tracking is performed by DeepSort Tracker from [deep_sort pytorch](https://github.com/ZQPei/deep_sort_pytorch).
<p align='center'>
<img src="images/res2.png" width="23%"></img>
<img src="images/res3.png" width="23%"></img>
<img src="images/res4.png" width="23%"></img>
<img src="images/res5.png" width="23%"></img> 
</p>

<h2>Performance</h2>
<p align='center'>
<img src="images/performance.png" width="70%"></img> 
</p>

<h2>Detection</h2>
<p>
Detection is performed by a wrap over of original github repo for paper [Yolo5Face: Why reinventing a face detector?](https://github.com/deepcam-cn/yolov5-face). This paper implements a yolov5n with wing loss, which is trained over WIDEFace dataset, to perform face detection.
Face detection with this repo was pretty smooth and robust to illumination changes, side poses and varying resolution of faces. 
<h4>Detection Only - QuickSetUp</h4>

  * det.py file performs detection only.
  * Change the camera device to webcam device id of your system on line 19 in det.py
  * You need to create a weights folder and put [this](https://drive.google.com/file/d/1oOxuQk6CN76pC02A2d1vWS2b35rKzdaA/view?usp=sharing) Yolov5 finetuned model in it.
  * Change device in det.py line 16, for gpu use gpu=0/1 and for cpu platform use gpu=-1.
  * Change size of the image to your desired value 320 or 480 on line 8 in det.py


```
python3 det.py
```
</p>

<h2>Quick SetUp</h2>

* main.py performs detection, recognition (of my team mates faces) and tracking.
* Pretrained models for [detection](https://drive.google.com/drive/folders/1DP99QP7pSuKZT8uSuBZQESFwhuInK3vc?usp=sharing) and tracking (Pretrained model already in repo) can be reused.
* For recogniton model you need to fine tune the model on custom faces to be recognized and get embeddings of the faces to be recognized and save in embeddings folder. For details check out Recognition subsection. Pre-trained recog for my team mates and sample embeddings are [here](https://drive.google.com/drive/folders/1DP99QP7pSuKZT8uSuBZQESFwhuInK3vc?usp=sharing)
* You are all set!

```
python3 main.py
```

<h2>Recogntion</h2>
<p>
  Recognition is performed using InceptionResNetv3, which is pretrained on VGGFace2, and we finetuned on faces of the people we wanted to recognize, in this case, we were three team members, whose faces we wanted algo to recognize, so we took 40 different images of each person, with varying backgrounds, clothes, illumination and face poses and sizes. In total, this model was fine tuned on 130 images roughly for 30 epochs.  
</p>
<p>
  Recogniton using above mentioned detector is performed as follows:
  
  * After fine tuning model, 3 different images of each person were passed though the model, one face from front, and other from left anf right side poses, and output of the final layer of trained Inception ResNet are stored in a file.
  * At time of recognition, the detected faces from yolov5 detector are cropped and saved.
  * These cropped faces are passed through this inception resnet to get face embeddings.
  * These embeddings are then compared with the saved embeddings using euclidean distannce. If embeddings of the faces in the frame does not cross similarity threshold person is declared as unnown otherwise, closest embedding are used to assign label.
  
 <h4>Detection and Recognition - QuickSetUp</h4>
 
 * det_recog.py performs detection and recognition only.
 * Perform all changes, in the det_recog.py, as mention in detection for det.py.
 * Train the inceptionNet model on few images of persons to be recognized and put in same 'weights' folder.
 * A pre-trained resnet model for 320 and 480 image size, to recognize faces of my team members is [here](https://drive.google.com/drive/folders/1DP99QP7pSuKZT8uSuBZQESFwhuInK3vc?usp=sharing)
 * Load approprate resnet pretrained model, according to the selected image size.
 * Create a folder, call it embeddings.
 * Pass atleast 2 sample images of ech persom from pretrained model, and save it in embeddings folder, sample embeddings folder for my team mates faces is [here](https://drive.google.com/drive/folders/1DP99QP7pSuKZT8uSuBZQESFwhuInK3vc?usp=sharing).
 
 
 
```
python3 det_recog.py
```
 
  <h2> Tracking </h2>
  <p> 
  Tracking is performed by [Deep SORT](https://github.com/nwojke/deep_sort) tracker, which is original SORT (Simple Online Real Time Tracking) tracker plus a deep convolutional network added to it, for improving the tracker's recognition accuracy. Simple SORT tracker uses just kalman filter based predicted state of the tracked object and does not includes any deep representation of the tracked item and hence leads to losing track of objcet for occlusion, or lightning changes. In our case as the side pose or illumination change or scale changes occured, the simple SORT tracker lost object track, and the track disappeared gradually whereas this deep SORT tracker performed really well and did not lose object track.
  </p>
  <p>
  DeepSORT tracker used here, has associated pre-trained convolutional network, for extracing deep representation of the tracked object to improve the mapping of tracked object with detector's detections. This convolutional network was pretrained on human body(pedestrians) not human faces but gave acceptabel performance when we used it for tracking faces.
  </p>
  <h4> Tracking and Detection Only- QuickSetUp</h4>
  
  * det_track.py performs detection and tracking only.
  * Perform all changes in det_track.py as performed in det.py for detection.
  
  ```
  python3 det_track.py
  ```
  
  <h2>HardWare SetUp</h2>
  Components used:
  
  * [Jetson Xavier NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-nx/)
  * [Logitech BRIO Webcam](https://www.logitech.com/en-us/products/webcams/brio-4k-hdr-webcam.960-001105.html)
  * USB3.0 to connect webcam to Xavier
  <p>This algorithm is ran on jetson Xavier NX connected to a logitech BRIO usb camera and results were displayed on local system by connecting an ssh tunnel using -X option to the xavier.</p>
  <h2>Acknowledgments</h2>
  
  * [Detection by Yolo5Face](https://github.com/elyha7/yoloface)
  * [Recognition by Recogn module of FaceNet PyTorch](https://github.com/timesler/facenet-pytorch#use-this-repo-in-your-own-git-project)
  * [Tracking from DeepSORT tracker of DeepSORT PyTorch](https://github.com/ZQPei/deep_sort_pytorch)
  

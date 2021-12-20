from imutils.video import VideoStream
import torch
import torch.nn as nn
import argparse
import imutils
import time
import numpy as np
import cv2
import torch.nn.functional as F
from face_detector import YoloDetector
from deep_sort import build_tracker
from utils.parser import get_config
from utils.draw import draw_boxes

############## INITIAL SETUP #############
#Set up device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
size = 320  # size = 320 or 480
use_cuda = torch.cuda.is_available()
print("USE CUDA: ",use_cuda)
print("DEVICE: ",device)



# image standardization
def standardize(image):
	image = (image - 127.5) / 128.0
	return image

def webcam_xavier(camera):
	cap=cv2.VideoCapture(camera)

	if not cap.isOpened():
		print('Failed to open camera')
		return None
	else:
		print("[INFO] starting video stream...")
		return cap

# if a video path was not supplied, grab the reference to the web cam
vs = webcam_xavier(0) #'/dev/video0')

################# MODELS ##################
# Detection model
model = YoloDetector(target_size=size, gpu=0, min_face=20)   # =>set gpu=0 for GPU =>set gpu=-1 for CPU

#Initialize tracker
cfg=get_config(config_file='deep_sort.yaml')
cfg.USE_FASTREID = False
cfg.USE_MMDET = False
deepsort = build_tracker(cfg, use_cuda=use_cuda)

results = []
frame_idx=0
# loop over frames from the video stream
while True:
	frame_idx+=1
	################## READ FRAMES ###################
	status,frame = vs.read()
	# resize the frame (so we can process it faster)
	w, h, c = frame.shape
	image = cv2.resize(frame, (size, size))

	################## DETECT AND RECOGNIZE ###################
	det_start=time.time()
	boxes, _ = model.predict(image) #detection
	det_time=time.time() - det_start
	if len(boxes)>0:
		################# TRACK ####################
		bbox_xywh=[]
		#Fromat detections in deepsort acceptable format
		for box in boxes:
			x1,y1,x2,y2=box
			# Scale back up face locations 
			box_w=x2-x1
			box_h=y2-y1
			xc=(x1+x2)/2
			yc=(y1+y2)/2
			det=[xc,yc,box_w,box_h]
			# create a new object tracker for the bounding box and add it
			# to our multi-object tracker
			bbox_xywh.append(det)
		bbox_xywh=np.array(bbox_xywh).reshape(len(bbox_xywh),4)	
		bbox_conf=[0.85]*len(bbox_xywh)
		track_start=time.time()
		tracking_output=np.array(deepsort.update(bbox_xywh,bbox_conf,image)).reshape(-1,5)
		print("Detections: ",bbox_xywh)
		print("Tracker Updated: ",tracking_output)
		track_time=time.time()-track_start

	# No Recognition/Det for this frame, use Kalman filter
	else:
		print("No dets for", str(frame_idx) ,"th frame!!!!")
		tracking_output=deepsort.update(None,None,image)
		track_time=0
		print("Tracker Update w/o dets: ",tracking_output)
	frame_time=det_time+track_time		
	text="FPS: "+str(np.round(1/frame_time,2))
	cv2.putText(frame, text, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	
	
	################# VISUALIZATION #################
	#VISUALIZE Tracking
	try:
		if len(tracking_output)>0:
			i=0
			bbox_xyxy = tracking_output[:, :4]
			identities = tracking_output[:, -1]
			#Rescale boxes
			for i,box in enumerate(bbox_xyxy):
				x1,y1,x2,y2 = [int(i) for i in box]
				x1 = int(x1 * h/size)
				y1 = int(y1 * w/size)
				x2 = int(x2 * h/size)
				y2 = int(y2 * w/size)
				bbox_xyxy[i]=[x1,y1,x2,y2]
			frame = draw_boxes(frame, bbox_xyxy, identities)
			
	except NameError:
		print("Tracker not initialized")
		pass	
	# Visualize detection results
	if len(boxes)>0:
		for idx,box in enumerate(boxes):
			
			#Draw bbox
			(x1, y1, x2, y2) = box
			x1 = int(x1 * h/size)
			y1 = int(y1 * w/size)
			x2 = int(x2 * h/size)
			y2 = int(y2 * w/size)
			cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
			#cv2.putText(frame, "Track-id: "+str(tid), (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
			
	# show the output frame
	cv2.imshow("Frame", frame)
	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

vs.release()
# close all windows
cv2.destroyAllWindows()

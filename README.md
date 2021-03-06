# Enhanced-YOLO

It's based on AlexeyAB's Windows fork of YOLOv3 https://github.com/AlexeyAB/darknet (version from 05.2020). It focuses on improving IoU and confidence of recognized objects, by applying Bayesian Inference to intermediate results obtained from YOLO.

Requirements:
  - OpenCV 3.4
  - Visual Studio 2017
  - Nvidia CUDA 10
  - Nvidia cuDNN 7
  - YOLO weights & YOLO configuration file
  - Python 3.7 & Open Images & Pandas

As a testing set, I used Open Images organized in following way:
  - C:\images\picture no. x\
  - C:\images\picture no. x\data\
  - C:\images\picture no. x\picture.jpg
  
PowerShell script helps with organizing photos, results and running everything.

If you already have YOLO, then only add/replace files that are in folder "just modified YOLO files".
	
I HIGHLY RECOMMEND using newer build of "darknet", because the one I used crashes from time to time when using GPU.

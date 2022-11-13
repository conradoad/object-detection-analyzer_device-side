from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
import numpy as np
from pyproftool import PyProfTool

# USE_VIDEO_PORT = False
# F_WIDTH = 2560
# F_HEIGHT = 1440
# F_WIDTH = 1920
# F_HEIGHT = 1080
# F_WIDTH = 1280
# F_HEIGHT = 720

USE_VIDEO_PORT = True
F_WIDTH = 800 #SVGA
F_HEIGHT = 600 #SVGA
# F_WIDTH = 640 #VGA/480p
# F_HEIGHT = 480 #VGA/480p
# F_WIDTH = 320 #QVGA/240p
# F_HEIGHT = 240 #QVGA/240p

frames_sample = 10

prof = PyProfTool("Camera Capture")
prof.enable()

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (F_WIDTH, F_HEIGHT)
camera.exposure_mode = 'sports'

rawCapture = PiRGBArray(camera, size=(F_WIDTH, F_HEIGHT))

# allow the camera to warmup
time.sleep(0.1)

frames_count = 0

images = []

camera.capture(rawCapture, use_video_port=USE_VIDEO_PORT, format="bgr")
rawCapture.truncate(0)
time.sleep(1)

dir = "./images_" + ("VIDEO_PORT" if USE_VIDEO_PORT else "STILL_PORT") + "_" + str(F_WIDTH) + "_X_" + str(F_HEIGHT) + "/"

try:
    os.mkdir(dir)    
except:
    print("Directory already exists") 
finally:
    os.chdir(dir)

prof.start_point("All captures")
# grab an image from the camera
while (frames_count < frames_sample):
    prof.start_point("One capture")
    camera.capture(rawCapture, use_video_port=USE_VIDEO_PORT, format="bgr")
    rawCapture.truncate(0)
    prof.end_point("One capture")
    cv2.imwrite("image_" + str(frames_count) + ".jpeg", rawCapture.array)
    time.sleep(0.5)
    frames_count += 1
prof.end_point("All captures")

prof.show()


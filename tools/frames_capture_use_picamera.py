from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from pyproftool import PyProfTool

F_WIDTH = 2560
F_HEIGHT = 1440
# F_WIDTH = 1920
# F_HEIGHT = 1080
# F_WIDTH = 1280
# F_HEIGHT = 720

frames_sample = 10

prof = PyProfTool("Camera Capture")
prof.enable()

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (F_WIDTH, F_HEIGHT)

rawCapture = PiRGBArray(camera, size=(F_WIDTH, F_HEIGHT))

# allow the camera to warmup
time.sleep(0.1)

frames_count = 0

images = []

prof.start_point("All captures")

# grab an image from the camera
while (frames_count < frames_sample):
    prof.start_point("One capture")
    camera.capture(rawCapture, use_video_port=False, format="rgb")
    images.append(rawCapture.array)
    rawCapture.truncate(0)
    frames_count += 1
    prof.end_point("One capture")
prof.end_point("All captures")

prof.show()
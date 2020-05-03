import cv2 as cv
import sys
import argparse
import numpy as np
import math
import time
from math import acos, degrees
from decimal import Decimal
import os.path

#Initializing parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 608       #Width of network's input image
inpHeight = 608      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Position finder using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.', type=str)
parser.add_argument('--video', help='Path to video file.', default="")
args = parser.parse_args()
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

# Classes file
classesFile = "coco.names" # Can add your own .names file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Finding angle function
def thetaCal(opposite, adjacent):
    """
    Pass opposite, adjacent of object to find the angle of object relative to some other object.
    """
    opposite = opposite * (-1)
    theta = math.atan2(opposite, adjacent)  # * (180 / 3.1415)
    theta = math.degrees(theta)
    theta = round(theta, 2)

    if theta < 0:
        theta = 180 + theta
        theta = theta + 180
        theta = round(theta, 2)
    return theta


# Mapping the angle to the position of object in frame or relative positions between objects
def relPos(thehta):
    if theta >= 340:
        theta = theta - 360
    elif theta >= -20 and theta <= 20:
        val = "Right"
    elif theta >= 20 and theta <= 70:
        val = "Top Right"
    elif theta >= 70 and theta <= 110:
        val = "Top"
    elif theta >= 110 and theta <= 160:
        val = "Top Left"
    elif theta >= 160 and theta <= 200:
        val = "Left"
    elif theta >= 200 and theta <= 250:
        val = "Bottom Left"
    elif theta >= 250 and theta <= 290:
        val = "Bottom"
    elif theta >= 290 and theta <= 340:
        val = "Bottom Right"
    else:
        val = "Unidentified Quadrant"
    return val

def objLocation(X, Y):
    """
    Use this code block if you want to fix camera/frame and have to find positon of objects
    in the grid of 3x3 cells. 
    """
    
    yGrid = frame.shape[0] / 3
    xGrid = frame.shape[1] / 3

    if X <= xGrid and Y <= yGrid:
        val = "Top Left"
    elif X <= xGrid and Y <= (yGrid + yGrid):
        val = "Middle Left"
    elif X <= xGrid and Y <= (yGrid + yGrid + yGrid):
        val = "Bottom Left"
    elif X <= xGrid + xGrid and Y <= (yGrid):
        val = "Top Middle"
    elif X <= xGrid + xGrid and Y <= (yGrid + yGrid):
        val = "Middle"
    elif X <= xGrid + xGrid and Y <= (yGrid + yGrid + yGrid):
        val = "Bottom Middle"
    elif X <= xGrid + xGrid + xGrid and Y <= (yGrid):
        val = "Top Right"
    elif X <= xGrid + xGrid + xGrid and Y <= (yGrid + yGrid):
        val = "Middle Right"
    elif X <= xGrid + xGrid + xGrid and Y <= (yGrid + yGrid + yGrid):
        val = "Bottom Right"
    else:
        val = "Unidentified block"
    return val

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    centerY = frame.shape[0] / 2
    centerX = frame.shape[1] / 2

    disX, disY = 0

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    objCenter = []
    disXY = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                disX0 = center_x - centerX
                disY0 = center_y - centerY
                disXY.append([disX0, disY0])
                objCenter.append([center_x, center_y])  # List to save center points of objects
                width = int(detection[2] * frameWidth) # Total width of object
                height = int(detection[3] * frameHeight) # Total Height of object
                left = int(center_x - width / 2) # on x axis top left corner of object
                top = int(center_y - height / 2) # on y x axis top right corner of object
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

# Process inputs
winName = 'Object Location Finding Using YOLO and OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)
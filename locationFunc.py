import math
import time
from math import acos, degrees
from decimal import Decimal

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
# Object Location finder

Knowing Object location in a frame is useful in real-time applications, in the repo, I have demonstrated how you can find locations of objects in a given frame. 


## Prerequisties 

- YOLO .names file
- YOLO .weights file
- YOLO .cfg file

## Usage 

- A single image:

`python3 object_detection_yolo.py --image=bird.jpg`

- A video file:

`python3 object_detection_yolo.py --video=run.mp4`

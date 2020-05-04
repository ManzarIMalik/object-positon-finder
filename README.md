# Object Location finder

Knowing Object location in a frame is useful in real-time applications, in the repo, I have demonstrated how you can find locations of objects in a given frame. 


## Prerequisties 

- YOLO .names file
- YOLO .weights file
- YOLO .cfg file

## Usage 

- A single image:

`python3 find.py --image=bird.jpg`

- A video file:

`python3 find.py --video=run.mp4`

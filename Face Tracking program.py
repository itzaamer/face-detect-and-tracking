#sources
#https://pyimagesearch.com/2018/07/30/opencv-object-tracking/
#https://github.com/twairball/face_tracking
import cv2
import sys, datetime
from time import sleep
import argparse
import numpy as np

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

class FaceDetector():
    """This is a class for detecing the face on frame"""
    def __init__(self, cascPath=r"C:\Users\user\Desktop\demo\haar cascade\haarcascade_frontalface_default.xml"):
        """
        The constructor for FaceDetector class.
  
        Parameters:
           path : path for the haarcascade_file""" 
        self.faceCascade = cv2.CascadeClassifier(cascPath)
    def detect(self, frame):
        """
        The function to detect on the frame.
  
        Parameters: frame of video"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=9)
        return faces

class FaceTracker():
    """This is a class for tracking the face on frame"""
    def __init__(self, frame, face):
        """
        The constructor for FaceTracker class."""
        (x,y,w,h) = face#co-ordinate for the bounding box
        self.face = (x,y,w,h)
        #Here we selecting tracker
        #self.tracker = cv2.TrackerCSRT_create()
        self.tracker = cv2.TrackerKCF_create()
        #self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.tracker.init(frame, self.face)# Initialize tracker with first frame and bounding box
    
    def update(self, frame):
        """
        The function to update the tracker"""
        _, self.face = self.tracker.update(frame)
        return self.face

def draw_boxes(frame, boxes, color=(0,255,0)):
    """This fucntion is used to draw the Bonding Box"""
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
    return frame

def resize_image(image, size_limit=500.0):
    """This fucntion is used to resize the image
    resize the frame (so we can process it faster) and grab the"""
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image

class Controller():
    """This is a class for controlling the time for frame Tracker"""
    def __init__(self, event_interval=6):
        self.event_interval = event_interval
        self.last_event = datetime.datetime.now()

    def trigger(self):
        """Return True if should trigger event"""
        return self.get_seconds_since() > self.event_interval
    
    def get_seconds_since(self):
        current = datetime.datetime.now()
        seconds = (current - self.last_event).seconds
        return seconds

    def reset(self):
        self.last_event = datetime.datetime.now()

class Pipeline():
    def __init__(self, event_interval=6):
        self.controller = Controller(event_interval=event_interval)    
        self.detector = FaceDetector()
        self.trackers = []
    
    def detect_and_track(self, frame):
        # get faces 
        faces = self.detector.detect(frame)

        # reset timer
        self.controller.reset()

        # get trackers
        self.trackers = [FaceTracker(frame, face) for face in faces]

        # return state = True for new boxes
        # if no faces detected, faces will be a tuple.
        new = type(faces) is not tuple

        return faces, new
    
    def track(self, frame):
        boxes = [t.update(frame) for t in self.trackers]
        # return state = False for existing boxes only
        return boxes, False
    
    def boxes_for_frame(self, frame):
        if self.controller.trigger():
            return self.detect_and_track(frame)
        else:
            return self.track(frame)



def run_face_tracker(event_interval=6):
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print('Cannot open video')
        sys.exit()
    
    # read first frame
    ok, frame = video_capture.read()
    if not ok:
        print('Error reading video')
        sys.exit()

    # init detection pipeline
    pipeline = Pipeline(event_interval=event_interval)

    # hot start detection
    # read some frames to get first detection
    faces = ()
    detected = False
    while not detected:
        _, frame = video_capture.read()
        faces, detected = pipeline.detect_and_track(frame)
        print("hot start; ", faces, type(faces), "size: ", np.array(faces).size)

    draw_boxes(frame, faces)
    
    while True:
        # Capture frame-by-frame
        ok, frame = video_capture.read()

        # update pipeline
        boxes, detected_new = pipeline.boxes_for_frame(frame)

        # logging means tracking events that happen when some run program
        state = "DETECTOR" if detected_new else "TRACKING"
        print("[%s] boxes: %s" % (state, boxes))

        # update screen
        color = GREEN if detected_new else BLUE
        draw_boxes(frame, boxes, color)

        # Display the resulting frame
        cv2.imshow('Video-Tracking', frame)

        ## if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def argsParser():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", "-i", type=int,action='store'
                        ,default=6,help='Detection interval in seconds, default=6')
    args = ap.parse_args()

    return args

def main():
    args = argsParser()
    run_face_tracker(args.interval)

# Run face tracking
if __name__ == '__main__':
    main()

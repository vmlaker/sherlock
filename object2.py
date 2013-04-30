"""Object detection pipeline."""

import multiprocessing
import datetime
import time
import sys
import cv2
import numpy as np

import sharedmem
import mpipe
import util
import iproc
import cascade

DEVICE   = int(sys.argv[1])
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DURATION = float(sys.argv[4])  # In seconds.

# Create a process-shared (common) table keyed on timestamps
# and holding references to allocated memory and other useful values.
manager = multiprocessing.Manager()
common = manager.dict()

class Preprocessor(mpipe.OrderedWorker):
    """First step of image processing."""
    def doTask(self, tstamp):
        """Return preprocessed image."""
        iproc.preprocess(
            common[tstamp]['image_in'], common[tstamp]['image_pre'])
        return tstamp
 
class Detector(mpipe.OrderedWorker):
    """Detects objects."""
    def __init__(self, classifier, color):
        self._classifier = classifier
        self._color = color

    def doTask(self, tstamp):
        """Run object detection."""
        result = list()
        try:
            image = common[tstamp]['image_pre']
            size = np.shape(image)[:2]
            rects = self._classifier.detectMultiScale(
                image,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=tuple([x/20 for x in size]),
                maxSize=tuple([x/2 for x in size]),
                )
            if len(rects):
                for a,b,c,d in rects:
                    result.append((a,b,c,d, self._color))
        except:
            print('Error in detector !!!')
        return result

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

class Postprocessor(mpipe.OrderedWorker):
    def doTask(self, (tstamp, rects,)):
        """Augment the input image with results of processing."""
        # Make a flat list from a list of lists .
        rects = [item for sublist in rects for item in sublist]

        # Draw rectangles.
        for x1, y1, x2, y2, color in rects:
            cv2.rectangle(
                common[tstamp]['image_in'],
                (x1, y1), (x1+x2, y1+y2),
                color=color,
                thickness=2,
                )

        # Write image dimensions and framerate.
        size = np.shape(common[tstamp]['image_in'])[:2]
        iproc.writeOSD(
            common[tstamp]['image_in'],
            ('%dx%d'%(size[1], size[0]),
             '%.2f, %.2f, %.2f fps'%framerate.tick()),
            )
        return tstamp

cv2.namedWindow('object detection 2', cv2.cv.CV_WINDOW_NORMAL)
class Viewer(mpipe.OrderedWorker):
    """Displays image in a window."""
    def doTask(self, tstamp):
        try:
            image = common[tstamp]['image_in']
            cv2.imshow('object detection 2', image)
            cv2.waitKey(1)
        except:
            print('Error in viewer !!!')
        return tstamp

class Staller(mpipe.OrderedWorker):
    def doTask(self, (tstamp, results,)):
        """Make sure the timestamp is at least a certain 
        age before propagating it further."""
        delta = datetime.datetime.now() - tstamp
        duration = datetime.timedelta(seconds=2) - delta
        if duration > datetime.timedelta():
            time.sleep(duration.total_seconds())
        return tstamp

# Create the detector pipelines.
detector_pipes = list()
for classi in cascade.classifiers:
    detector_pipes.append(
        mpipe.Pipeline(
            mpipe.Stage(
                Detector, 1, 
                classifier=classi, 
                color=cascade.colors[classi])))

# Assemble the image processing pipeline:
#
#            detector_pipe(s)                   viewer
#                  ||                             ||
#   preproc --> detector --> postproc --+--> filter_viewer --> staller
#
preproc = mpipe.Stage(Preprocessor)
detector = mpipe.FilterStage(detector_pipes, max_tasks=1)
postproc = mpipe.Stage(Postprocessor)
pipe_viewer = mpipe.Pipeline(mpipe.Stage(Viewer))
filter_viewer = mpipe.FilterStage((pipe_viewer,), max_tasks=2)
staller = mpipe.Stage(Staller)
preproc.link(detector)
detector.link(postproc)
postproc.link(filter_viewer)
filter_viewer.link(staller)
pipe_iproc = mpipe.Pipeline(preproc)

# Create an auxiliary process (modeled as a one-task pipeline)
# that simply pulls results from the image processing pipeline
# and deallocates the associated memory.
def deallocate(task):
    for tstamp in pipe_iproc.results():
        del common[tstamp]
pipe_dealloc = mpipe.Pipeline(mpipe.UnorderedStage(deallocate))
pipe_dealloc.put(True)  # Start it up right away.

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Run the video capture loop, allocating shared memory
# and feeding the image processing pipeline.
now = datetime.datetime.now()
end = now + datetime.timedelta(seconds=DURATION)
while end > now:

    # Mark the timestamp. This is the index by which 
    # image procesing stages will access allocated memory.
    now = datetime.datetime.now()

    # Capture the image.
    hello, image = cap.read()

    # Allocate shared memory for a copy of the input image.
    shape = np.shape(image)
    dtype = image.dtype
    image_in = sharedmem.empty(shape, dtype)
    image_pre = sharedmem.empty(shape[:2], dtype)

    # Copy the input image to its shared memory version.
    image_in[:] = image.copy()
    
    # Add to the common table.
    common[now] = {
        'image_in'  : image_in,   # Input image.
        'image_pre' : image_pre,  # Preprocessed image.
        }

    # Put the timestamp on the image processing pipeline.
    pipe_iproc.put(now)

# Capturing of video is done. Now let's shut down all
# pipelines by sending them the "stop" task.
pipe_iproc.put(None)
for pipe in detector_pipes:
    pipe.put(None)
    for result in pipe.results():
        pass
pipe_viewer.put(None)
for result in pipe_viewer.results():
    pass

# Signal deallocator to stop and wait until it frees all memory.
pipe_dealloc.put(None)
for result in pipe_dealloc.results():
    pass

# The end.

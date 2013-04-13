"""Multiple process image processing using shared memory."""

import multiprocessing
import datetime
import sys
import cv2
import numpy as np

import sharedmem
import mpipe
import util
import iproc

DEVICE   = int(sys.argv[1])
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DURATION = float(sys.argv[4])  # In seconds.

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Create a process-shared (common) table keyed on timestamps
# and holding references to allocated memory and other useful values.
manager = multiprocessing.Manager()
common = manager.dict()

# Accumulation of thresholded differences.
image_acc = None  

# Keep track of previous iteration's timestamp.
tstamp_prev = None  

def step1(tstamp):
    """Return preprocessed image."""
    global tstamp_prev
    alpha, tstamp_prev = iproc.getAlpha(tstamp_prev)
    
    # Reassign the modified object to the proxy container in order to
    # notify manager that the mutable value (the dictionary) has changed. See for details:
    # http://docs.python.org/library/multiprocessing.html#multiprocessing.managers.SyncManager.list
    itstamp = common[tstamp]
    itstamp['alpha'] = alpha
    common[tstamp] = itstamp  # Reassign modified object to proxy.

    iproc.preprocess(common[tstamp]['image_in'], common[tstamp]['image_pre'])
    return tstamp
 
class Step2Worker(mpipe.OrderedWorker):
    def doTask(self, tstamp):
        """Compute difference between given image and accumulation,
        then accumulate and set result with the difference. 
        Initialize accumulation if needed (if opacity is 100%.)"""
        
        image_pre = common[tstamp]['image_pre']
        image = common[tstamp]['image_in']
        alpha = common[tstamp]['alpha']
        
        # Use global accumulation.
        global image_acc

        # Initalize accumulation if so indicated.
        if alpha == 1.0:
            image_acc = np.empty(np.shape(image_pre))

        # Compute difference.
        cv2.absdiff(
            image_acc.astype(image_pre.dtype),
            image_pre,
            common[tstamp]['image_diff']
            )

        self.putResult(tstamp)

        # Accumulate.
        hello = cv2.accumulateWeighted(
            image_pre,
            image_acc,
            alpha,
            )
        
def step3(tstamp):
    """Postprocess image using given difference."""
    iproc.postprocess(
        common[tstamp]['image_in'], 
        common[tstamp]['image_diff'],
        common[tstamp]['image_out'], 
        )
    return tstamp

class Viewer(mpipe.OrderedWorker):
    """Base class viewer implementation, specialized in
    subclasses by overriding getName()."""
    def doTask(self, tstamp):
        name = self.getName()
        try:
            cv2.namedWindow(name, cv2.cv.CV_WINDOW_NORMAL)
            image = common[tstamp][name]
            cv2.imshow(name, image)
            cv2.waitKey(1)
        except:
            print('error running viewer %s !!!'%name)
        return tstamp
    def getName(self): return 'base'

class ViewerIn(Viewer):
    def getName(self): return 'image_in'
class ViewerPre(Viewer):
    def getName(self): return 'image_pre'
class ViewerDiff(Viewer):
    def getName(self): return 'image_diff'
class ViewerOut(Viewer):
    def getName(self): return 'image_out'

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

def printStatus(tstamp):
    """Print the framerate to stdout."""
    print('%05.3f, %05.3f, %05.3f'%framerate.tick())
    return tstamp

# Create the image processing stages.
step1 = mpipe.OrderedStage(step1)
step2 = mpipe.Stage(Step2Worker)
step3 = mpipe.OrderedStage(step3)

# Create the output display stages.
viewer_in = mpipe.Stage(ViewerIn)
viewer_pre = mpipe.Stage(ViewerPre)
viewer_diff = mpipe.Stage(ViewerDiff)
viewer_out = mpipe.Stage(ViewerOut)
printer = mpipe.OrderedStage(printStatus)

# Link the stages into a pipeline.
#
#  step1 ---+---> step2 ------+---> step3 --------+---> viewer_out
#           |                 |                   |
#           +---> viewer_in   +---> viewer_diff   +---> printer
#           |
#           +---> viewer_pre
#
step1.link(step2)
step1.link(viewer_in)
step1.link(viewer_pre)
step2.link(step3)
step2.link(viewer_diff)
step3.link(viewer_out)
step3.link(printer)
pipe = mpipe.Pipeline(step1)

def pull(tstamp):
    prev = None
    for tstamp in pipe.results():
        if prev is not None:
            del common[prev]
        prev = tstamp
    del common[prev]
pipe2 = mpipe.Pipeline(mpipe.UnorderedStage(pull))
pipe2.put(True)

now = datetime.datetime.now()
end = now + datetime.timedelta(seconds=DURATION)
while end > now:
    now = datetime.datetime.now()
    hello, image = cap.read()

    # Allocate shared memory for
    #   a copy of the input image,
    #   the preprocessed image,
    #   the diff image,
    #   the resulting output image.
    shape = np.shape(image)
    dtype = image.dtype
    image_in   = sharedmem.empty(shape,     dtype)
    image_pre  = sharedmem.empty(shape[:2], dtype)
    image_diff = sharedmem.empty(shape[:2], dtype)
    image_out  = sharedmem.empty(shape,     dtype)
    
    # Copy the input image to it's shared memory version,
    # and also to the eventual output image memory.
    image_in[:] = image.copy()
    image_out[:] = image.copy()
    
    common[now] = {
        'image_in'   : image_in,
        'image_pre'  : image_pre,
        'image_diff' : image_diff,
        'image_out'  : image_out,
        'alpha'      : 1.0,
        }
    pipe.put(now)

pipe.put(None)
pipe2.put(None)

for result in pipe2.results():
    pass

# The end.

Sherlock
========

OpenCV in Python, with multiprocessing and shared memory.

This is a collection of small codes for processing live video 
(like from a webcam) with [OpenCV](http://opencv.org).
Image data (NumPy arrays) are shared by multiple processes
using [numpy-sharedmem Python module](http://bitbucket.org/cleemesser/numpy-sharedmem).
Parallel processing workflow is implemented in 
[MPipe framework](http://vmlaker.github.io/mpipe/concepts.html). 

Installation
------------

### 1. Get the Sherlock codes

```
git clone --recursive http://github.com/vmlaker/sherlock
```

### 2. Get OpenCV Python bindings

Using YUM:
```
yum install opencv-python
```
Or using Aptitude:
```
aptitude install python-opencv
```

### 3. Get numpy-sharedmem

```
hg clone http://bitbucket.org/cleemesser/numpy-sharedmem
cd numpy-sharedmem
python setup.py install --user
```

### 4. Get MPipe

```
pip install --user mpipe
```

Playback test
-------------

First thing, test your OpenCV Python bindings.
The following command shows live view from the first video device (i.e. ``/dev/video0``) for a duration of 5 seconds:
```
python playcv2.py 0 640 480 5
```

Motion detection
----------------

The following programs demonstrate simplified motion detection.
The codes increase in complexity with changes intended to enhance performance. 

* ``diffavg1.py`` - single process
* ``diffavg2.py`` - add multiprocessing
* ``diffavg3.py`` - add shared memory
* ``diffavg4.py`` - add filtering

Processing algorithm is a run-of-the-mill foreground/background segmentation using scene average. 
Run the first program on input from ``/dev/video0`` with:
```
python diffavg1.py 0 640 480 5
```
Now try the others -- compare the framerates and any lag in video output. 
You might consider profiling resource usage by running with ``time``.


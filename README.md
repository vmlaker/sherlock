Sherlock
========

OpenCV in Python, with multiprocessing and shared memory!

A collection of code doing image processing of live video (e.g. from a webcam),
demonstrating performance enhancements of full utilization of SMP 
systems (i.e. multi-core workstations) by combining the power of
multiprocessing and shared memory. All that in Python, of course!

Software dependencies
---------------------

Before you can run the codes, have the following libraries installed for your Python interpreter:

### OpenCV

[Open Source Computer Vision](http://opencv.org) is used for image processing algorithms. 
On a YUM system, install the Python bindings with:
```
yum install opencv-python
```
or if using Aptitude, try:
```
aptitude install python-opencv
```

### numpy-sharedmem

Images (large NumPy arrays) are efficiently accessed by multiple processes using the ``sharedmem`` module. 
Take a look at the [numpy-sharedmem project](http://bitbucket.org/cleemesser/numpy-sharedmem) for details.
Install the module with:
```
hg clone https://cleemesser@bitbucket.org/cleemesser/numpy-sharedmem/
cd numpy-sharedmem
python setup.py install --user
```

### MPipe

Multiprocessing workflow is implemented in the [MPipe framework](http://vmlaker.github.io/mpipe/concepts.html). 
There's a number of [ways to install MPipe](http://vmlaker.github.io/mpipe/download.html) but the easiest is probably:
```
pip install --user mpipe
```

Playback test
-------------

First thing, test your OpenCV Python bindings with simple video playback. 
The following command shows live view from the first video device (i.e. ``/dev/video0``) for a duration of 5 seconds:
```
python playcv2.py 0 640 480 5
```

Motion detection
----------------

The following programs demonstrate simplified motion detection:

* ``diffavg1.py`` - single process
* ``diffavg2.py`` - add multiprocessing
* ``diffavg3.py`` - add shared memory
* ``diffavg4.py`` - adjust for mid-stage result
* ``diffavg5.py`` - add frame drop

The codes increase in complexity with changes intended to enhance performance. 

The overall processing algorithm is a simple run-of-the-mill foreground/background segmentation using scene average. 
For example, to run the first program using ``/dev/video0``, try this:
```
python diffavg1.py 0 640 480 5
```
Now try the others -- compare the framerates and any lag in video output. 
You might consider profiling resource usage by running with ``time``. 
 
Have fun!

.. image:: http://vmlaker.github.io/sherlock/logo.png
  :alt: Sherlock Logo

Sherlock
========

OpenCV in Python, with multiprocessing and shared memory.

A collection of small codes for processing live video 
(like from a webcam) with `OpenCV <http://opencv.org>`_.
Image data (NumPy arrays) are shared by multiple processes using
`numpy-sharedmem Python module <http://bitbucket.org/cleemesser/numpy-sharedmem>`_.
Parallel processing workflow is implemented with 
`MPipe <http://vmlaker.github.io/mpipe/concepts.html>`_. 

Installation
------------

1. Get OpenCV Python bindings, using YUM:
::

   yum install opencv-python
 
or using Aptitude:
::

   aptitude install python-opencv

2. Get the project code:
::

   git clone --recursive http://github.com/vmlaker/sherlock

3. Run make:
::

   cd sherlock
   make

Playback test
-------------

First thing, test your OpenCV Python bindings.
The following command shows live view from the first video device 
(i.e. ``/dev/video0``) for a duration of 8 seconds:
::

   make playcv2

Motion detection
----------------

The following demonstrate simplified motion detection.
Each iteraton increases in complexity with cumulative changes 
intended to enhance performance. 

1) Run one process:
::

   make diffavg1

2) Add parallel processing:
::

   make diffavg2

3) Add shared memory:
::

   make diffavg3

4) Add filtering:
::

   make diffavg4

Processing algorithm is a run-of-the-mill foreground/background segmentation using scene average. 
Consider profiling resource usage by running with ``time``.

Object detection
----------------

Run face detection using Haar cascade classifier:
::

   make object1
   make object2

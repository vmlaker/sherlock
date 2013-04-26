"""Capture webcam video using GStreamer, and playback."""

import time
import sys

import numpy as np
import cv2
import gst

import util

DEVICE   = sys.argv[1]
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DEPTH    = int(sys.argv[4])
DURATION = float(sys.argv[5])

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

# Create the display window.
title = 'playing GStreamer capture'
cv2.namedWindow(title, cv2.cv.CV_WINDOW_NORMAL)

def onVideoBuffer(pad, idata):
    """Convert buffer data and show as image."""
    image = np.ndarray(
        shape=(HEIGHT, WIDTH, DEPTH), 
        dtype=np.uint8, 
        buffer=idata,
        )
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, image)
    cv2.waitKey(1)

    # Print the framerate.
    print('%.2f, %.2f, %.2f'%framerate.tick())

# Assemble the GStreamer video stream pipeline.
specs = [
    ('source','v4l2src'          ,[('device', DEVICE), ]),
    ('color' ,'ffmpegcolorspace' ,[] ),
    ('scale' ,'videoscale'       ,[] ),
    ('filter','capsfilter', [
            ('caps', 
             'video/x-raw-rgb,width=%s,height=%s,bpp=%s'%(
                    WIDTH, HEIGHT, DEPTH*8)),
            ]),
    ('fake', 'fakesink', []),
    ]
(vpipe, elements, args,) = util.create_gst_pipeline(specs)

# Add the buffer probe on the last element.
pad = next(elements['fake'].sink_pads())
pad.add_buffer_probe(onVideoBuffer)

# Run the video pipeline for designated duration.
vpipe.set_state(gst.STATE_PLAYING)
time.sleep(DURATION)
vpipe.set_state(gst.STATE_NULL)

# The end.

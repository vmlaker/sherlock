"""Image processing routines."""

import datetime
import cv2
import numpy as np

def getAlpha(tstamp_prev, max_life=1.0):
    """Return (alpha, tstamp_new) based on given timestamp.
    The alpha value is in range [0.0, 1.0], scaled from the
    distance between *tstamp_prev* and now, the distance 
    maximized at *max_life* seconds.
    For example:  distance  max_life  alpha
                  --------  --------  -----
                      3         6      0.5
                      6         6      1.0
                      9         6      1.0
    """
    now = datetime.datetime.now()
    alpha = 1.0  # Default is 100% opacity.
    if tstamp_prev:
        # alpha = min {delta_t, max_life} / max_life        
        tdelta = now - tstamp_prev
        alpha = tdelta.total_seconds()
        alpha = min(alpha, float(max_life))
        alpha /= max_life
    return alpha, now

def preprocess(image_in, image_out=None):
    """Turn to grayscale, then equalize histogram."""
    image_out = cv2.cvtColor(
        image_in,
        cv2.COLOR_BGR2GRAY, 
        image_out,
        )
    image_out = cv2.equalizeHist(
        image_out,
        image_out,
        )
    return image_out

def threshold(image_in, image_out=None):
    """Apply a sensible threshold."""
    hello, image_out = cv2.threshold(
        image_in,
        thresh=35,
        maxval=255,
        type=cv2.THRESH_BINARY,
        dst=image_out,
        )
    
    if 0: image_out = cv2.adaptiveThreshold(
        image_in,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=7,
        C=4,
        dst=image_out
        )
    return image_out
    
def postprocess(image, image_source, image_out=None, rect=False):
    """Augment *image* with contours from *image_source*.
    Operate inplace on *image*, unless given *output image*."""

    # Find contours.
    contours, hier = cv2.findContours(
        image_source,  # Note: findContours() changes the image.
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
        )        

    # Sort and filter contours.
    # Use a sensible threshold value based on image resolution.
    area_threshold = image_source.shape[0] * image_source.shape[1]
    area_threshold *= 0.00005 /2
    contours = sorted(
        contours, 
        key=lambda x: cv2.contourArea(x), 
        reverse=True)
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Since contours are sorted, we can safely break out 
        # of the loop once area falls below threshold.
        if area < area_threshold:
            break

        # Add this contour to the collection.
        filtered.append(contour)

    if image_out is None:
        image_out = image

    # Augment output image with contours.
    cv2.drawContours(
        image_out,
        contours=filtered,
        contourIdx=-1,
        color=(63, 200, 63),  # Green.
        thickness=2,
        )

    # Augment output image with rectangles.
    if rect: 
        for contour in filtered:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(
                image_out,
                (x,y),
                (x+w,y+h),
                color=(0, 255, 0),
                thickness=2,
                )

def writeOSD(image, lines, size=0.04):
    """Write text given in *lines* iterable, 
    the height of each line determined by *size* as
    proportion of image height."""

    # Compute row height at scale 1.0 first.
    (letter_width, letter_height), baseline = cv2.getTextSize(
        text='I', 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        thickness=1)

    # Compute actual scale to match desired height. 
    image_height = np.shape(image)[0]
    line_height = int(image_height * size)
    scale = float(line_height) / letter_height

    # Deterimine base thickness, based on scale.
    thickness = int(scale * 4)

    # Increase line height, to account for thickness.
    line_height += thickness * 3

    # Iterate the lines of text, and draw them.
    xoffset = int(letter_width * scale)
    yoffset = line_height
    for line in lines:
        cv2.putText(  # Draw the drop shadow.
            image,
            text=line,
            org=(xoffset+max(1, thickness/2), yoffset+max(1, thickness/2)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale,
            color=(0, 0, 0),
            thickness=thickness,
            )
        cv2.putText(  # Draw the text body.
            image,
            text=line,
            org=(xoffset, yoffset),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale,
            color=(215, 215, 70),
            thickness=thickness,
            )
        cv2.putText(  # Draw the highlight.
            image,
            text=line,
            org=(xoffset-max(1, thickness/3), yoffset-max(1, thickness/3)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale,
            color=(245, 255, 200),
            thickness=thickness/3,
            )
        yoffset += line_height
        
# The end.

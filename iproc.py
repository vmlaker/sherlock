"""Image processing routines."""

import datetime
import cv2

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
        tdelta = now - tstamp_prev
        alpha = tdelta.total_seconds()
        alpha = min(alpha, float(max_life))
        alpha /= max_life
    return alpha, now

def preprocess(image_in, image_out):
    """Equalize the grayscale version of input image,
    operate inplace on given output image."""
    cv2.cvtColor(
        image_in,
        cv2.COLOR_BGR2GRAY, 
        image_out,
        )
    cv2.equalizeHist(
        image_out,
        image_out,
        )

def preprocess2(image_in):
    """Equalize the grayscale version of input image.
    Return resulting image."""
    image_gray = cv2.cvtColor(
        image_in,
        cv2.COLOR_BGR2GRAY, 
        )
    image_out = cv2.equalizeHist(
        image_gray,
        )
    return image_out

def postprocess(image, image_diff, image_out=None):
    """Augment given image with given difference.
    Operate inplace on image, unless given output image."""

    # Apply threshold.
    hello, image_thresh = cv2.threshold(
        image_diff,
        thresh=35,
        maxval=255,
        type=cv2.THRESH_BINARY,
        )

    # Find contours.
    contours, hier = cv2.findContours(
        image_thresh,
        #np.copy(image_thresh),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
        )        

    # Sort and filter contours.
    # Use a sensible threshold value based on image resolution.
    area_threshold = image_thresh.shape[0] * image_thresh.shape[1]
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
        filtered,
        -1,
        color=(0, 255, 255),  # Yellow.
        thickness=2,
        )

    # Augment output image with rectangles.
    for contour in filtered:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(
            image_out,
            (x,y),
            (x+w,y+h),
            color=(0, 255, 0),
            thickness=2,
            )

# The end.

"""Classifier cascades."""

import os 
import cv2

# List of directories that are searched for classifier files.
dirs = (

    # On Fedora.
    '/usr/share/OpenCV/haarcascades',
    '/usr/share/OpenCV/lbpcascades',

    # On Debian.
    '/usr/share/opencv/haarcascades',
    '/usr/share/opencv/lbpcascades',

    # On OS X.
    '/usr/local/share/OpenCV/haarcascades',
    '/usr/local/share/OpenCV/lbpcascades',
)

# Listed below are classifiers used 
# (file names sans file extension.)
# Color values are in B,G,R format.
specs = (
    # ===== Face =====
    ('haarcascade_frontalface_alt2'    , (0, 255, 0)),
    #('haarcascade_frontalface_alt_tree', (0, 255, 0)),
    #('haarcascade_frontalface_alt'     , (0, 255, 0)),
    #('haarcascade_frontalface_default' , (0, 255, 0)),
    #('haarcascade_profileface'         , (0, 255, 0)),
    ('lbpcascade_frontalface'          , (0, 255, 0)),

    # ===== Ear, mouth, nose =====
    #('haarcascade_mcs_leftear'         , (127, 255, 255)),
    #('haarcascade_mcs_rightear'        , (127, 255, 255)),
    #('haarcascade_mcs_mouth'           , (255, 127, 127)),
    #('haarcascade_mcs_nose'            , (127, 127, 255)),

    # ===== Eye =====
    #('haarcascade_eye_tree_eyeglasses', (0, 255, 223)),
    ('haarcascade_eye'                , (0, 255, 223)),
    #('haarcascade_lefteye_2splits'    , (0, 255, 223)),
    #('haarcascade_righteye_2splits'   , (0, 255, 223)),
    #('haarcascade_mcs_eyepair_big'    , (0, 255, 223)),
    #('haarcascade_mcs_eyepair_small'  , (0, 255, 223)),
    #('haarcascade_mcs_lefteye'        , (0, 255, 223)),
    #('haarcascade_mcs_righteye'       , (0, 255, 223)),

    # ===== Body =====
    #('haarcascade_fullbody'           , (0, 255, 0)),
    #('haarcascade_lowerbody'          , (0, 255, 0)),
    #('haarcascade_mcs_upperbody'      , (0, 255, 0)),
    #('haarcascade_upperbody'          , (0, 255, 0)),
)

# OpenCV cascade classifiers and the colors table.
classifiers = list()
colors = dict()

# Process each classifier XML file in the configuration
# by iterating specs and directories.
for spec in specs:
    for dir in dirs:

        # Assemble the filename.
        full_path = os.path.join(dir, spec[0] + '.xml')
        
        # Check validity of the XML file by creating
        # OpenCV cascade classifier and testing its validity
        # (skip if invalid.)
        cfer = cv2.CascadeClassifier(full_path)
        if cfer.empty():
            continue

        # Add to the module containers.
        classifiers.append(cfer)
        colors[cfer] = spec[1]

if not classifiers:
    print('*** Warning: No classifers configured. ***')

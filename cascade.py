"""Classifier cascades."""

import os 
import cv2

path_to_haar = '/usr/share/OpenCV/haarcascades'
path_to_lbp = '/usr/share/OpenCV/lbpcascades'

specs = (
    # ===== Face =====
    #('haarcascade_frontalface_alt2.xml'    , path_to_haar, (0, 255, 0)),
    #('haarcascade_frontalface_alt_tree.xml', path_to_haar, (0, 255, 0)),
    #('haarcascade_frontalface_alt.xml'     , path_to_haar, (0, 255, 0)),
    ('haarcascade_frontalface_default.xml' , path_to_haar, (127, 255, 127)),
    ('haarcascade_profileface.xml'         , path_to_haar, (127, 255, 127)),
    #('lbpcascade_frontalface.xml'          , path_to_lbp,  (0, 255, 0)),

    # ===== Ear, mouth, nose =====
    ('haarcascade_mcs_leftear.xml'         , path_to_haar, (127, 127, 255)),
    ('haarcascade_mcs_rightear.xml'        , path_to_haar, (127, 127, 255)),
    #('haarcascade_mcs_mouth.xml'           , path_to_haar, (127, 255, 255)),
    ('haarcascade_mcs_nose.xml'            , path_to_haar, (255, 127, 127)),

    # ===== Eye =====
    #('haarcascade_eye_tree_eyeglasses.xml', path_to_haar, (0, 255, 0)),
    ('haarcascade_eye.xml'                , path_to_haar, (127, 0, 255)),
    #('haarcascade_lefteye_2splits.xml'    , path_to_haar, (0, 255, 0)),
    #('haarcascade_righteye_2splits.xml'   , path_to_haar, (0, 255, 0)),
    #('haarcascade_mcs_eyepair_big.xml'    , path_to_haar, (0, 255, 0)),
    #('haarcascade_mcs_eyepair_small.xml'  , path_to_haar, (0, 255, 0)),
    #('haarcascade_mcs_lefteye.xml'        , path_to_haar, (0, 255, 0)),
    #('haarcascade_mcs_righteye.xml'       , path_to_haar, (0, 255, 0)),

    # ===== Body =====
    #('haarcascade_fullbody.xml'           , path_to_haar, (0, 255, 0)),
    #('haarcascade_lowerbody.xml'          , path_to_haar, (0, 255, 0)),
    #('haarcascade_mcs_upperbody.xml'      , path_to_haar, (0, 255, 0)),
    #('haarcascade_upperbody.xml'          , path_to_haar, (0, 255, 0)),
)

# OpenCV cascade classifiers and the colors table.
classifiers = list()
colors = dict()
for spec in specs:
    full_path = os.path.join(spec[1], spec[0])
    cfer = cv2.CascadeClassifier(full_path)
    classifiers.append(cfer)
    colors[cfer] = spec[2]

# The end.

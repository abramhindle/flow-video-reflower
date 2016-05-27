# coding=utf8
import sys
import cv2
import numpy as np
from numpy import *
import liblo
import random
import time

def current_time():
    return int(round(time.time() * 1000))

target = liblo.Address(57120)

def send_img(img):
    h,w = img.shape
    l = [int(x) for x in img[0,0:w]]
    liblo.send(target, "/stream", *l)
    

fullscreen = False
cv2.namedWindow("remapped", cv2.WND_PROP_FULLSCREEN)

kinect = None
cap = None
if len(sys.argv) < 2:
    print "Opening vtest.avi"
    cap = cv2.VideoCapture("vtest.avi")
else:
    print "Opening %s" % sys.argv[1]
    cap = cv2.VideoCapture(sys.argv[1])

ret, frame1 = cap.read()
remapped = frame1
h,w,_ = frame1.shape

handlers = {
}

def handle_keys():
    global fullscreen
    global handlers
    k = cv2.waitKey(1000/60) & 0xff
    if k == 27:
        return True
    elif k == ord('f'):
        if not fullscreen:
            cv2.setWindowProperty("remapped", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
            fullscreen = True
        else:
            cv2.setWindowProperty("remapped", cv2.WND_PROP_FULLSCREEN, 0)
            fullscreen = False
    else:
        if k in handlers:
            handlers[k]()
    return False

starttime = None
fps=30
framesecond = 1000 / fps
myframes = 0
skips=0
frames = 0

while(1):
    # deal with slow video
    ret, frame2 = cap.read()
    myframes += 1
    mytime = current_time()
    if (starttime == None):
        starttime = current_time()
    expectedframes = (mytime - starttime) / framesecond
    #print "%s" % [mytime - starttime, expectedframes, myframes]
    while (expectedframes >  myframes):
        skips += 1
        print "Skipping a frame %s %s %s" % (expectedframes, myframes, 1.0*skips/myframes )
        ret, frame2 = cap.read()
        myframes += 1
    frame_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    gray_slice = frame_gray[h/2:h/2+1, 0:w]
    gray_slice = cv2.resize(gray_slice, (100,1))
    send_img(gray_slice)
    cv2.line(frame2, (0,h/2-1),(w,h/2-1), (255,0,0), 1)
    cv2.line(frame2, (0,h/2+1),(w,h/2+1), (255,0,0), 1)
    cv2.imshow('frame2',frame2)
    if handle_keys():
        break
    frames = frames + 1

cap.release()
cv2.destroyAllWindows()

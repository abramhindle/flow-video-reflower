# coding=utf8
import sys
import cv2
import numpy as np
from numpy import *
import freenect

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

if len(sys.argv) == 3:
    print "No Kinect: Opening %s" % sys.argv[2]
    kinect = cv2.VideoCapture(sys.argv[2])
    
scale = 0.25
levels = 6
winsize = 16
iterations = 3
polyn = winsize
polysigma = 2

"""
Grabs a depth map from the Kinect sensor and creates an image from it.
http://euanfreeman.co.uk/openkinect-python-and-opencv/
"""
def get_depth_map():    
    depth, timestamp = freenect.sync_get_depth()
 
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
 
    return depth



def get_kinect_video():    
    global kinect
    if kinect == None:
        print "Opening Kinect"
        kinect = cv2.VideoCapture(-1)
    ret, frame2 = kinect.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    return next



scaledown = 0.4
ret, frame1 = cap.read()
remapped = frame1

DEPTH_MAP = 0


prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
oldest = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

prvs = cv2.resize(prvs, (0,0), fx=scaledown, fy=scaledown) 
frame1 = cv2.resize(frame1, (0,0), fx=scaledown, fy=scaledown) 
#hsv = np.zeros_like(frame1)
#hsv[...,1] = 255
hsv = None

prvs = None

# todo 
#  make a matrix of points like flow but just the actual locations in the matrix
#  (0,0) (0,1) (0,2)
#  (1,0) (1,1) (1,2)
#  ...
#  Then add that matrix to the flow matrix
#  Then call remap
#
#  To work in larger spaces:
#  make the big point matrix
#  resize the flow matrix 
#  scale the flow matrix (e.g. if we did flow at 1/4 then multiple flow by 4)
#  add the matrices
#  remap

# todo
# get the kinect in here

#ptpts = ndarray((prvs.shape[0],prvs.shape[1],2))
ptpts = ndarray((remapped.shape[0],remapped.shape[1],2))
def mkpoints(ptpts):
    (w,h,n) = ptpts.shape
    for y in range(0,h):
        for x in range(0,w):
            ptpts[x,y,0] = y
            ptpts[x,y,1] = x
    return ptpts.astype('float32')

def init_ptpts(ptpts):
    return mkpoints(ptpts)

def reflow(flow,pts):
    return flow + pts

def reflow_resize(flow,pts):
    rflow = cv2.resize(flow, (pts.shape[1],pts.shape[0]))
    return (6.0/scaledown)*rflow + pts

class Reflow:    
    def __init__(self, pts):
        self.pts = pts
    def reflow(self,flow):
        rflow = cv2.resize(flow, (self.pts.shape[1],self.pts.shape[0]))
        return (6.0/scaledown)*rflow + self.pts

class ReflowDecay(Reflow):
    def __init__(self, pts, decay = 0.9, multiplier = 6.0):
        Reflow.__init__(self, pts)
        self.decay = decay
        self.history = None
        self.multiplier = multiplier

    def reflow(self,flow):
        rflow = cv2.resize(flow, (self.pts.shape[1],self.pts.shape[0]))
        if (self.history == None):
            self.history = rflow
        old = (self.multiplier/scaledown)*rflow + self.decay * self.history
        self.history = old
        return old + self.pts

ptpts = init_ptpts(ptpts)

frames = 0

reflower = ReflowDecay(ptpts,decay=0.9,multiplier=4)

while(1):
    ret, frame2 = cap.read()
    #depth_map = get_depth_map()
    depth_map = get_kinect_video()

    #next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #next = cv2.resize(next, (0,0), fx=scaledown, fy=scaledown) 

    next = cv2.resize(depth_map, (0,0), fx=scaledown, fy=scaledown)
    if prvs == None:
        prvs = cv2.resize(depth_map, (0,0), fx=scaledown, fy=scaledown)
        hsv = np.zeros((prvs.shape[0],prvs.shape[1],3))
        hsv[...,1] = 255
        hsv = hsv.astype('uint8')


    cv2.imshow('next',next)
    flow = cv2.calcOpticalFlowFarneback(prvs,next,scale,levels,winsize,iterations,polyn,polysigma,0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    rflow = reflower.reflow(flow)
    # BORDER_TRANSPARENT
    # BORDER_REPLICATE
    remapped = cv2.remap(remapped, rflow[...,0],rflow[...,1], 0, borderMode=cv2.BORDER_REFLECT_101 )#cv2.INTER_LINEAR)
    cv2.imshow('remapped',remapped)
    cv2.imshow('rgb',rgb)

    cv2.imshow('dept_map',depth_map)

    k = cv2.waitKey(1000/60) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    elif k == ord('f'):
        if not fullscreen:
            cv2.setWindowProperty("remapped", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
            fullscreen = True
        else:
            cv2.setWindowProperty("remapped", cv2.WND_PROP_FULLSCREEN, 0)
            fullscreen = False

    oldest = prvs
    prvs = next
    remapped = frame2
    frames = frames + 1


cap.release()
cv2.destroyAllWindows()

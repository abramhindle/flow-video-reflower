# coding=utf8
import sys
import cv2
import numpy as np
from numpy import *
import freenect
import liblo
import random
import time

def current_time():
    return int(round(time.time() * 1000))

target = liblo.Address(57120)

class Floater:
    def __init__(self, id,x,y,mx,my,color=(0,0,0),target=None,weight=1.0):
        self.id = id
        self.x = x
        self.y = y
        self.mx = mx
        self.my = my
        self.color = color
        self.target = target
        self.weight = weight

    def modx(self,x):
        x = x % self.mx
        self.x = self.x + (x - self.x)/self.weight

    def mody(self,y):
        y = y % self.my
        self.y = self.y + (y - self.y)/self.weight


    def mod(self,x,y):
        self.modx(x)
        self.mody(y)

    def dmod(self,dx,dy):
        self.mod(self.x + dx, self.y + dy)

    def update(self):
        self.send()

    def send(self):
        if not (self.target == None):
            liblo.send(target, "/flow", int(self.id), float(self.x), float(self.y),float(self.mx),float(self.my),float(self.weight))

    def repel(self, floater):
        if abs(floater.x - self.x) <= 2.0 and abs(floater.x - self.x) <= 2.0:
            self.mod( self.x + random.gauss(0.0,floater.weight),  self.y + random.gauss(0.0,floater.weight))
        else:
            self.dmod( (floater.weight*5.0) / (self.x - floater.x + 0.0001), (floater.weight * 5.0) / (self.y - floater.y + 0.0001))


def floater_repel(floaters):
    if (len(floaters) == 1):
        return
    for i in range(0,len(floaters) - 1):
        for j in range(i,len(floaters)):
            floaters[i].repel(floaters[j])
            floaters[j].repel(floaters[i])

def draw_floaters(floaters, buffer):
    for floater in floats:        
        cv2.circle(buffer, (int(floater.x), int(floater.y)), int(10*floater.weight), floater.color, thickness=10, lineType=8)

def flow_floaters( floats, rflow ):
    for floater in floats:        
        (x,y) =  rflow[int(floater.y),int(floater.x),...] 
        floater.mod( x , y )
        floater.update()

        

floaterid = 1
    


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
freenect_use = False
def get_kinect_video():    
    if freenect_use == False or  not kinect == None:
        return get_kinect_video_cv()
    depth, timestamp = freenect.sync_get_video()  
    if (depth == None):
        return None
    return depth[...,1]



def get_kinect_video_cv():    
    global kinect
    if kinect == None:
        print "Opening Kinect"
        kinect = cv2.VideoCapture(-1)
    ret, frame2 = kinect.read()
    if not ret:
        return None
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

reflower = ReflowDecay(ptpts,decay=0.99,multiplier=1)
mx=remapped.shape[1]
my=remapped.shape[0]
colors = [
    (255,0,0),
    (0,255,0),
    (0,0,255)
]
weights = [1,2,4]
def mkFloater(x=None,y=None,c=None,weight=1.0):
    global floaterid
    global remapped
    global target
    mmx=remapped.shape[1]
    mmy=remapped.shape[0]
    if (x == None):
        x = mmx/2
    if (y == None):
        y = mmy/2
    floater = Floater(floaterid, x, y, mmx, mmy, colors[floaterid % len(colors)], target, weight=weights[floaterid % len(weights)])
    floaterid += 1
    return floater

n = 0
#floats = [mkFloater(i*mx/n,i*my/n) for i in range(0,n)]

def mkFlowHandler(decay=None, mult=None):
    global reflower
    def setter():
        if not decay == None:
            reflower.decay = decay
        if not mult == None:
            reflower.multiplier = mult
    return setter

handlers = {
    ord('1'): mkFlowHandler(decay=0.1),
    ord('2'): mkFlowHandler(decay=0.5),
    ord('4'): mkFlowHandler(decay=0.7),
    ord('5'): mkFlowHandler(decay=0.9),
    ord('6'): mkFlowHandler(decay=0.95),
    ord('7'): mkFlowHandler(decay=0.99),
    ord('8'): mkFlowHandler(decay=0.995),
    ord('9'): mkFlowHandler(decay=0.999),
    ord('0'): mkFlowHandler(decay=0.9999),
    ord('q'): mkFlowHandler(mult=0.1),
    ord('w'): mkFlowHandler(mult=0.5),
    ord('e'): mkFlowHandler(mult=0.7),
    ord('r'): mkFlowHandler(mult=0.7),
    ord('t'): mkFlowHandler(mult=1.0),
    ord('y'): mkFlowHandler(mult=2.0),
    ord('u'): mkFlowHandler(mult=4.0),
    ord('i'): mkFlowHandler(mult=8.0),
    ord('o'): mkFlowHandler(mult=16.0)
}

def handle_keys():
    global fullscreen
    global handlers
    k = cv2.waitKey(1000/60) & 0xff
    if k == 27:
        return True
    elif k == ord('s'):
        for i in xrange(0,30*5):
            ret, frame2 = cap.read()
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
while(1):
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
        



    #ret, frame2 = cap.read()

    #depth_map = get_depth_map()
    depth_map = get_kinect_video()
    if depth_map == None:
        print "Bad?"
        continue
    depth_map = cv2.flip(depth_map, 1)


    #next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #next = cv2.resize(next, (0,0), fx=scaledown, fy=scaledown) 

    next = cv2.resize(depth_map, (0,0), fx=scaledown, fy=scaledown)
    if prvs == None:
        prvs = cv2.resize(depth_map, (0,0), fx=scaledown, fy=scaledown)
        hsv = np.zeros((prvs.shape[0],prvs.shape[1],3))
        hsv[...,1] = 255
        hsv = hsv.astype('uint8')


    cv2.imshow('next',next)
    #flow = next
    #flow = cv2.calcOpticalFlowFarneback(prvs,next,scale,levels,winsize,iterations,polyn,polysigma,0)
    #print flow.dtype
    #print next.shape
    #print prvs.shape
    #print prvs.dtype
    #flow = np.empty((next.shape[0],next.shape[1],2), dtype=np.float32)
    #flow = np.dstack([next.astype(np.float32),prvs.astype(np.float32)])
    flow = np.reshape([next.astype(np.float32),prvs.astype(np.float32)],(next.shape[0],next.shape[1],2),order='F')
    print flow.shape
    #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    #hsv[...,0] = ang*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    rflow = reflower.reflow(flow)
    # BORDER_TRANSPARENT
    # BORDER_REPLICATE
    remapped = cv2.remap(remapped, rflow[...,0],rflow[...,1], 0, borderMode=cv2.BORDER_REFLECT )#cv2.INTER_LINEAR)
    (rh,rw,_) = remapped.shape

    #flow_floaters(floats, rflow)
    #floater_repel( floats )
    #draw_floaters( floats, remapped )

    cv2.imshow('remapped',remapped)
    #cv2.imshow('rgb',rgb)
    cv2.imshow('dept_map',depth_map)

    if handle_keys():
        break


    oldest = prvs
    prvs = next
    remapped = frame2
    frames = frames + 1


cap.release()
cv2.destroyAllWindows()

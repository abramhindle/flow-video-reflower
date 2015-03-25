# coding=utf8
import cv2
import numpy as np
from numpy import *
cap = cv2.VideoCapture("vtest.avi")
scaledown = 0.2
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
oldest = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
remapped = oldest
prvs = cv2.resize(prvs, (0,0), fx=scaledown, fy=scaledown) 
frame1 = cv2.resize(frame1, (0,0), fx=scaledown, fy=scaledown) 
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

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

ptpts = ndarray((prvs.shape[0],prvs.shape[1],2))
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

ptpts = init_ptpts(ptpts)
frames = 0
while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    next = cv2.resize(next, (0,0), fx=scaledown, fy=scaledown) 
    # Python: cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])   flow
    #    Parameters:	
    #
    #        prev – first 8-bit single-channel input image.
    #        next – second input image of the same size and the same type as prev.
    #        flow – computed flow image that has the same size as prev and type CV_32FC2.
    #        pyr_scale – parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
    #        levels – number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
    #        winsize – averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
    #        iterations – number of iterations the algorithm does at each pyramid level.
    #        poly_n – size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
    #        poly_sigma – standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
    #        flags – 
    #flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2)
    scale = 0.25
    levels = 6
    winsize = 16
    iterations = 3
    polyn = winsize
    polysigma = 2
    flow = cv2.calcOpticalFlowFarneback(prvs,next,scale,levels,winsize,iterations,polyn,polysigma,0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    rflow = reflow(flow,ptpts)
    remapped = cv2.remap(remapped, rflow[...,0],rflow[...,1], 0)#cv2.INTER_LINEAR)
    cv2.imshow('frame2',remapped)
    cv2.imshow('rgb',rgb)
    cv2.imshow('next',next)
    cv2.imshow('oldest',oldest)
    #cv2.imshow('frame2',rgb)
    #cv2.imshow('frame2',frame2)
    k = cv2.waitKey(1000/60) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next
    frames = frames + 1
    if frames % 30 == 0:
        oldest = prvs
        remapped = oldest
cap.release()
cv2.destroyAllWindows()

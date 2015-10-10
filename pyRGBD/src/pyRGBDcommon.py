'''
Created on Aug 26, 2015

@author: francisco
'''
import cv2
import numpy as np
from pyRGBDlib import *


def getPointCloudFromRangeAsso(assoFn,i,step=10):
    fnImg,fnDep=assoFn[i]
    img1 = cv2.imread(fnBase+'/'+fnImg)
    dep1 = cv2.imread(fnBase+'/'+fnDep,cv2.IMREAD_ANYDEPTH)
    pf,cf,pts,clr=getPointCloudFromRange(dep1,img1,step)
    return pf,cf,pts,clr,img1,dep1

def get2DGridOrganizedFromAsso(assoFn,i,step=10):
    fnImg,fnDep=assoFn[i]
    img1 = cv2.imread(fnBase+'/'+fnImg)
    dep1 = cv2.imread(fnBase+'/'+fnDep,cv2.IMREAD_ANYDEPTH)
    frame=get2DGridOrganizedPointCloud(dep1,img1,step)
    return frame

def getNormalPlaneAndCurvature(pts):
    meaPts=np.mean(pts,0)
    cenPts=pts-meaPts
    covMat=cenPts.transpose().dot(cenPts)/pts.shape[0]
    w, u, vt = cv2.SVDecomp(covMat)
    #print "w",w
    #print "u",u
    #print "vt",vt
    l=np.sqrt(w)
    minL=l[2]
    ro=minL/np.sum(l)
    nv=u[:,2]
    #print "nv",nv
    #print "ro",ro
    if nv.dot([0,0,-1])>0:
        nv=-nv
    return nv,ro

def isPlane(pts):
    minPts=np.min (pts,0)
    maxPts=np.max (pts,0)
    # Z=0 mean invalid point
    if minPts[2]==0:
        #print "Invalid"
        return False
    # Z distane >10 long distane edge
    if abs(maxPts[2]-minPts[2])>0.8:
        #print maxPts[2]-minPts[2]
        #print "long distance"
        return False
    nv,ro=getNormalPlaneAndCurvature(pts)
    if ro<0.06:
        return False
    return True
    
fnBase='/media/francisco/Packard Bell/Users/paco/Pictures/datasets/RGBD/cvg/rgbd_dataset_freiburg1_desk'
#fnBase='/media/francisco/Packard Bell/Users/paco/Pictures/datasets/RGBD/cvg/rgbd_dataset_freiburg1_xyz'
fnAsso=fnBase+'/association.txt'

with open(fnAsso) as f:
    asso = f.readlines()
assoFn=[] #array of associated file name tuples (rgb,depth)
for l in asso:
    ts1,fnRgb,ts2,fnDep=l.rstrip().split(" ")
    assoFn.append((fnRgb,fnDep))


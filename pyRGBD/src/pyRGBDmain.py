#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 19/12/2014

@author: Francisco Dominguez
'''
import cv2
import myICP as icp
from kdtree import KDTree
from visual import points,arrow,vector,rate
import numpy as np
from pyHotDraw.Images.pyHImageFilters import FundamentalMatrix
from pyRGBDlib import *
#from __main__ import time


def getPointCloudFromRangeAsso(assoFn,i,step=10):
    fnImg,fnDep=assoFn[i]
    img1 = cv2.imread(fnBase+'/'+fnImg)
    dep1 = cv2.imread(fnBase+'/'+fnDep,cv2.IMREAD_ANYDEPTH)
    pf,cf,pts,clr=getPointCloudFromRange(dep1,img1,step)
    return pf,cf,pts,clr,img1,dep1

def centerPoints(p):
    m=np.mean(p,0)
    return p-m
    
#fnBase='/media/francisco/Packard Bell/Users/paco/Pictures/datasets/RGBD/cvg/rgbd_dataset_freiburg1_desk'
fnBase='/media/francisco/Packard Bell/Users/paco/Pictures/datasets/RGBD/cvg/rgbd_dataset_freiburg1_xyz'
fnAsso=fnBase+'/association.txt'

with open(fnAsso) as f:
    asso = f.readlines()
assoFn=[] #array of associated file name tuples (rgb,depth)
for l in asso:
    ts1,fnRgb,ts2,fnDep=l.rstrip().split(" ")
    assoFn.append((fnRgb,fnDep))

ptsTf,clrTf,ptsT,clrT,imgT,depT=getPointCloudFromRangeAsso(assoFn,0,1)
ptsMf,clrMf,ptsM,clrM,imgM,depM=getPointCloudFromRangeAsso(assoFn,1,1)
points(pos=ptsT,size=1,color=clrT)   
# TT=np.eye(4,4)
# ptsTT=ptsT
# clrTT=clrT
# for i in range(1):
#     ptsMf,clrMf,ptsM,clrM,imgM,depM=getPointCloudFromRangeAsso(assoFn,i+1)
#     ptsM=icp.transformDataSetUsingTransform(ptsM,TT)
#     print "ICP start %d" % i
#     #Tf,pitM=icp.fitICPkdTree(ptsM,ptsT)
#     Tf,pitM=icp.fitICP(ptsM,ptsT)
#     #points(pos=pitM,size=4,color=clrM)
#     ptsT=pitM
#     TT=TT.dot(Tf)
#     pitMf=icp.transformDataSetUsingTransform(ptsMf,TT)
#     ptsTT=np.vstack((ptsTT,pitMf))
#     clrTT=np.vstack((clrTT,clrMf))
# points(pos=centerPoints(ptsTT),size=2,color=clrTT)#center and display points
# arrow(pos=(0,0,0), axis=(1,0,0), shaftwidth=0.01,color=(1,0,0))
# arrow(pos=(0,0,0), axis=(0,1,0), shaftwidth=0.01,color=(0,1,0))
# arrow(pos=(0,0,0), axis=(0,0,-1), shaftwidth=0.01,color=(0,0,1))
# arrow(pos=(0,0,0), axis=(0.1,0.1,-0.1), shaftwidth=0.01,color=(1,1,0))
fm=FundamentalMatrix()
fm.imgcv1=imgT
fm.imgcv2=imgM
#cv2.imshow("imagent",imgT)
#cv2.imshow("imagenm",fm.imgcv2)
img1=fm.process()
F,pt1,pt2=fm.data
pt1i=np.int32(pt1)
pt2i=np.int32(pt2)

dT=np.array([piInv(imgT,depT,x,y) for (x,y) in pt1i])
ptT,clT=dT[:,0,:],dT[:,1,:]
points(pos=ptT,size=3,color=(0,1,0))
print ptT.shape

dM=np.array([piInv(imgM,depM,x,y) for (x,y) in pt2i])
ptM,clM=dM[:,0,:],dM[:,1,:]
#points(pos=ptM,size=3,color=(0,0,1))
print ptM.shape

print "ptM",icp.distData(ptM,ptT)

Tcv,piM=icp.fitICP(ptM, ptT)
print Tcv

piMa=icp.transformDataSetUsingTransform(ptsM, Tcv)
points(pos=piMa,size=1,color=clrM)#center and display points
points(pos=piM,size=5,color=(1,0,0))

print "piM",icp.distData(piM,ptT)
#cv2.imshow("Fm1",img1)

# Tim,imM=icp.fitICP(ptsM,ptsT)
# points(pos=imM,size=1,color=clrM)#center and display points
# 
# print Tim
# print imM.shape,ptsT.shape
# ms=min(imM.shape[0],ptsT.shape[0])
# e=imM[:ms]-ptsT[:ms]
# print e.shape
# dImg=np.sum(e**2)**(0.5)
# print dImg.shape,dImg
while True:
    rate(10)      
    k = cv2.waitKey(10)
    if not k == -1:
        break
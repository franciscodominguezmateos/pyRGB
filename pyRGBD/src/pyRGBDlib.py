#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on Aug 25, 2015

@author: francisco
'''
import numpy as np
import math
import cv2

def getRotMatX(a):
    G=np.eye(4,4)
    c=math.cos(a)
    s=math.sin(a)
    G[1,1]=c
    G[1,2]=s
    G[2,1]=-s
    G[2,2]=c 
    return np.matrix(G)   
def getRotMatY(a):
    G=np.eye(4,4)
    c=math.cos(a)
    s=math.sin(a)
    G[0,0]=c
    G[0,2]=-s
    G[2,0]=s
    G[2,2]=c
    return np.matrix(G)
def getRotMatZ(a):
    G=np.eye(4,4)
    c=math.cos(a)
    s=math.sin(a)
    G[0,0]=c
    G[0,1]=s
    G[1,0]=-s
    G[1,1]=c
    return np.matrix(G)
def getEulerMat(a,b,c):
    return getRotMatX(a)*getRotMatY(b)*getRotMatY(c)

def getPyr(pd1,levels=4):
    p1=[]
    p1.append(pd1)
    for i in range(levels-1):
        pd1=np.float32(cv2.pyrDown(pd1))
        p1.append(pd1)
    return p1

def centerPoints(p):
    m=np.mean(p,0)
    return p-m

def piInv(Ic,Iz,u,v,level=1):
    #Intrinsic data for Stum dataset
    fx = 525.0/level  # focal length x
    fy = 525.0/level  # focal length y
    cx = 319.5/level  # optical center x
    cy = 239.5/level  # optical center y
    factor = 5000.0 # for the 16-bit PNG files
    deep=float(Iz[v,u])
    Z=deep/factor
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    r=float(Ic[v,u,2])/255.0
    g=float(Ic[v,u,1])/255.0
    b=float(Ic[v,u,0])/255.0
    p=(X,-Y,-Z)
    return (p,(r,g,b))

def piI(Id,u,v,level=1):
    '''
    return a 3d point from Iage depth and u,v unprojected
    '''
    #Intrinsic data for Stum dataset
    fx = 525.0/level  # focal length x
    fy = 525.0/level  # focal length y
    cx = 319.5/level  # optical center x
    cy = 239.5/level  # optical center y
    factor = 5000.0 # for the 16-bit PNG files
    deep=float(Id[v,u])
    Z=deep/factor
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    p=(X,-Y,-Z)
    return p

def piD(X,Y,Z,level=1):
    '''
    return a 2d point from a 2D project
    '''
    Y=-Y
    Z=-Z+1e-8
     #Intrinsic data for Stum dataset
    fx = 525.0/level  # focal length x
    fy = 525.0/level  # focal length y
    cx = 319.5/level  # optical center x
    cy = 239.5/level  # optical center y
    x=X*fx/Z+cx
    y=Y*fy/Z+cy
    z=1.0
    return (x,y,z)

def getPointCloudFromRange(dep,img,step=10):
    width =img.shape[1]
    height=img.shape[0]
    pts=[]
    ptsFull=[]
    clr=[]
    clrFull=[]
    for u in range(width):
        for v in range(height):
            p,c=piInv(img,dep,u,v)
            if -p[2]>0:# Z=-p[2]
                ptsFull.append(p)
                clrFull.append(c)
                if(u % step==0 and v % step==0):
                    pts.append(p)
                    clr.append(c)
    return (np.array(ptsFull),np.array(clrFull),np.array(pts),np.array(clr))

def get2DGridOrganizedPointCloud(dep,img,step=10):
    width =img.shape[1]
    height=img.shape[0]
    rows=height/step
    cols=width /step
    frame=[]
    for j in range(0,height,step):
        frame.append([])
        #print j
        for i in range(0,width,step):
            frame[j/step].append([])
            for jp in range(step):
                for ip in range(step):
                    u=i+ip
                    v=j+jp
                    p,c=piInv(img,dep,u,v)
                    x=i/step
                    y=j/step
                    frame[y][x].append((p,c))
                    #print i,j,y,x,len(frame[y][x])
    return frame
    

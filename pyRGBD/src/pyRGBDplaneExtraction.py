'''
Created on Aug 25, 2015

@author: Francisco Dominguez
'''
import cv2
import math
from visual import points,arrow
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
    meaPt=np.mean(pts,0)
    cenPts=pts-meaPt
    covMat=cenPts.transpose().dot(cenPts)#/pts.shape[0]
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
    return nv,ro,meaPt

def isPlane(pts):
    minPts=np.min (pts,0)
    maxPts=np.max (pts,0)
    # Z=0 mean invalid point
    if minPts[2]==0:
        #print "Invalid"
        return False
    # Z distane >10 long distane edge
    if abs(maxPts[2]-minPts[2])>0.5:
        #print "long distance"
        #print maxPts[2]-minPts[2]
        return False
    nv,ro,mn=getNormalPlaneAndCurvature(pts)
    #print ro
    if ro<0.06:
        print "ro",ro
        return False
    return True

def normalDist(vn0,vn1):
    return vn0.dot(vn1)

def planeDist((pts0,pt0,vn0,ro0),(pts1,pt1,vn1,ro1)):
    pass
def distPlanePoint((pts0,pt0,vn0,ro0),pt):
    d =vn0.dot(pt0)
    dx=vn0.dot(pt)
    return dx-d
#0 big plane 1 new littleplane
def nearPlanes((pts0,pt0,vn0,ro0),(pts1,pt1,vn1,ro1)):
    d=distPlanePoint((pts0,pt0,vn0,ro0),pt1)
    #print "d",d
    if np.abs(d)>0.01:
        return False
    nd=normalDist(vn0,vn1)
    #print "nd",nd
    if nd<0.6:
        return False
    dif10=pts1-pt0
    dif01=pts0-pt1
    d10=np.linalg.norm(dif10,axis=1)
    d01=np.linalg.norm(dif01,axis=1)
    md10=np.min(d10)
    md01=np.min(d01)
    dif=pt1-pt0
    di2=dif.dot(dif)
    #s=pts0.shape[0]+pts1.shape[0]
    d=np.sqrt(di2)
    difP=d-md10-md01
    #print "difP",difP
    if abs(difP)>0.01:
        return False
    return True

def fusePlanes((pts0,pt0,vn0,ro0),(pts1,pt1,vn1,ro1)):
    pts=np.append(pts0,pts1,0)
    #print "fuse",pts.shape
    vn,ro,mn=getNormalPlaneAndCurvature(pts)
    return pts,mn,vn,ro
    
fnBase='/media/francisco/Packard Bell/Users/paco/Pictures/datasets/RGBD/cvg/rgbd_dataset_freiburg1_desk'
#fnBase='/media/francisco/Packard Bell/Users/paco/Pictures/datasets/RGBD/cvg/rgbd_dataset_freiburg1_xyz'
fnAsso=fnBase+'/association.txt'

with open(fnAsso) as f:
    asso = f.readlines()
assoFn=[] #array of associated file name tuples (rgb,depth)
for l in asso:
    ts1,fnRgb,ts2,fnDep=l.rstrip().split(" ")
    assoFn.append((fnRgb,fnDep))

ptsTf,clrTf,ptsT,clrT,imgT,depT=getPointCloudFromRangeAsso(assoFn,150,10)
meanPtsT=np.mean(ptsTf,0)
points(pos=ptsTf-meanPtsT,size=1,color=clrTf)   
print ptsT.shape,ptsTf.shape

p=get2DGridOrganizedFromAsso(assoFn,150,5)
npf=np.array(p)
print npf.shape
npfp=npf[:,:,:,0,:]
npfc=npf[:,:,:,1,:]
print "npfp",npfp.shape
pts=np.vstack(npfp[:,:,0])
print "pts",pts.shape
#points(pos=pts,size=2,color=(1,0,0))
npp=np.array(npfp)
ptsPlane=[]
ptsNp=[]
arrPlane=[]
planes=[]
for x in range(npfp.shape[1]):
    for y in range(npfp.shape[0]):
        if isPlane(npp[y,x]):
            nv,ro,mnPt=getNormalPlaneAndCurvature(npp[y,x])
            ptsPlane.append(mnPt)
            #print "ro",ro
            arrPlane.append(nv)
            planes.append((npp[y,x],mnPt,nv,ro))
        else:
            ptsNp.append(npp[y,x,0])
points(pos=ptsNp-meanPtsT,size=4,color=(1,0,0))
#for i,(p,vn) in enumerate(zip(ptsPlane-meanPtsT,arrPlane)):
#    if i%8==0:
#        arrow(pos=p,axis=vn/50,color=(0,1,0))        
    

colors=[(0,0,0.5),(0,0.5,0),(0,0.5,0.5),(0.5,0,0),(0.5,0,0.5),(0.5,0.5,0),(0.5,0.5,0.5),
        (0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1),
        (0.5,0.5,1),(0.5,1,0.5),(0.5,1,1),(1,0.5,0.5),(1,0.5,1),(1,1,0.5)]

print "planes",len(planes)
i=0
planesOut=[]
while planes!=[]:
    pln0=planes.pop()
    planesNew=[]
    while planes!=[]:
        pln1=planes.pop()
        if nearPlanes(pln0,pln1):
            pln0=fusePlanes(pln0,pln1)
        else:
            planesNew.append(pln1)
    planesOut.append(pln0)
    p=pln0
    points(pos=p[0]-meanPtsT,size=1,color=colors[i%20])
    if p[0].shape[0]>1000:
        arrow (pos=p[1]-meanPtsT,axis=p[2]/10,color=colors[i%20])
    else:    
        arrow (pos=p[1]-meanPtsT,axis=p[2]/40,color=colors[i%20])    
    i+=1
    planes=planesNew
    print "pn",len(planesNew)

print "planesOut",len(planesOut)
bigPlanes=[]
ltPlanes=[]
for i,p in enumerate(planesOut):
    if p[0].shape[0]>1000:
        bigPlanes.append(p)
        #points(pos=p[0]-meanPtsT,size=1,color=colors[i%20])
        #arrow (pos=p[1]-meanPtsT,axis=p[2]/10,color=colors[i%20])
        print "p",p[0].shape[0],p[3]
    else:
        ltPlanes.append(p)
# for i,p in enumerate(ltPlanes):
#     print "lp",p[0].shape[0],p[3]
    
print "bp",len(bigPlanes)

print "THE END"

if __name__ == '__main__':
    pass
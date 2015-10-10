'''
Created on Sep 3, 2015

@author: Francisco Dominguez
'''
from pyRGBDcommon import *
import cv2
import numpy as np
import numpy.linalg as la

# import cudamat as cm
# 
# cm.cublas_init()
# 
# # create two random matrices and copy them to the GPU
# a = cm.CUDAMatrix(np.random.rand(32, 256))
# b = cm.CUDAMatrix(np.random.rand(256, 32))
# 
# # perform calculations on the GPU
# c = cm.dot(a, b)
# d = c.sum(axis = 0)
# 
# # copy d back to the host (CPU) and print
# print(d.asarray())

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
def JI(I2c):
    '''
    return two float gray images of gradient of I2c
    '''
    gray = I2c #cv2.cvtColor(I2c, cv2.COLOR_BGR2GRAY)
    I2x=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    I2y=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    return I2x,I2y

def getAb(Ixx,Ixy,Iyy,Itx,Ity):
    sIxx=np.sum(Ixx)
    sIxy=np.sum(Ixy)
    sIyy=np.sum(Iyy)
    sItx=np.sum(Itx)
    sIty=np.sum(Ity)
    A=np.array([[sIxx,sIxy],
                [sIxy,sIyy]])
    b=-np.array([[sItx],[sIty]])
    return A,b

ptsTf,clrTf,ptsT,clrT,imgT,depT=getPointCloudFromRangeAsso(assoFn,0,1)
ptsMf,clrMf,ptsM,clrM,imgM,depM=getPointCloudFromRangeAsso(assoFn,10,1)

#Color to Gray float32
imgTg=np.float32(cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY))/255.0
imgMg=np.float32(cv2.cvtColor(imgM, cv2.COLOR_BGR2GRAY))/255.0
pyrI1=getPyr(imgTg)
pyrI2=getPyr(imgMg)
level=2
I1=cv2.GaussianBlur(pyrI1[level],(3,3),0)
I2=cv2.GaussianBlur(pyrI2[level],(3,3),0)
Ix,Iy=JI(I2)
It=I1-I2
cv2.imshow("Ix",Ix)
cv2.imshow("Iy",Iy)
cv2.imshow("It",It)
Ixx=Ix*Ix
Iyy=Iy*Iy
Ixy=Ix*Iy
#Itt=It*It
Itx=It*Ix
Ity=It*Iy
cellShape=10
bIxx=blockshaped(Ixx,cellShape,cellShape)
bIxy=blockshaped(Ixy,cellShape,cellShape)
bIyy=blockshaped(Iyy,cellShape,cellShape)
bItx=blockshaped(Itx,cellShape,cellShape)
bIty=blockshaped(Ity,cellShape,cellShape)

i=0
hOf=I1.shape[0]/cellShape
wOf=I1.shape[1]/cellShape
print hOf,wOf
print bIxx.shape,bIxx[1].shape
I1c=cv2.cvtColor(I1, cv2.COLOR_GRAY2BGR)
off=cv2.calcOpticalFlowFarneback(I1,I2, 0.0, 3, 15, 3, 5, 1.2, 0)
print off.shape
for i in range(bIxx.shape[0]):
    A,b=getAb(bIxx[i],bIxy[i],bIyy[i],bItx[i],bIty[i])
    if la.det(A)<0.001:
        print "Matriz Singular"
    else:
        v=la.solve(A,b)
        r=i/wOf*cellShape+cellShape/2
        c=i%wOf*cellShape+cellShape/2
        vx=v[0]*cellShape/2
        vy=v[1]*cellShape/2
        print i,r,c,v.T,off[r,c]
        cv2.rectangle(I1c,(c-cellShape/2,r-cellShape/2),(c+cellShape/2,r+cellShape/2),(0,255,0))
        cv2.line(I1c,(c,r),(c+vx,r+vx),(0,0,255))
        #xoff=int(c+off[r,c,1]*cellShape*1e5)
        #yoff=int(r+off[r,c,0]*cellShape*1e5)
        #cv2.line(I1c,(c,r),(xoff,yoff),(255,0,0))
cv2.imshow("I1",I1c)

while True:
    #rate(10)      
    k = cv2.waitKey(10)
    if not k == -1:
        break


if __name__ == '__main__':
    pass
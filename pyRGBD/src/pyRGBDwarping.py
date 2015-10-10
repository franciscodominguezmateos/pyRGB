'''
Created on Aug 26, 2015

@author: francisco
'''

from pyRGBDcommon import *
import numpy.linalg as la
from pySophus import se3
from __builtin__ import False, True

def inShape(shape,x,y):
    xMax=shape[1]
    yMax=shape[0]
    if x>=xMax:
        return False
    if x<0:
        return False
    if y>=yMax:
        return False
    if y<0:
        return False
    return True
    
def pointTransform(G,X,Y,Z):
    r11=G[0,0]
    r12=G[0,1]
    r13=G[0,2]
    tx =G[0,3]
    r21=G[1,0]
    r22=G[1,1]
    r23=G[1,2]
    ty =G[1,3]
    r31=G[2,0]
    r32=G[2,1]
    r33=G[2,2]
    tz =G[2,3]
    Xp=r11*X+r12*Y+r13*Z+tx
    Yp=r21*X+r22*Y+r23*Z+ty
    Zp=r31*X+r32*Y+r33*Z+tz
    return Xp,Yp,Zp
    
def warpPoint(Id,G,u,v):
    '''
    We need just coordinates not color
    '''
    level=640/Id.shape[1]#pyrLevel
    #print "level",level,Ic.shape[1]
    #unproject
    (X,Y,Z)=piI(Id,u,v,level)
    if Z==0:#bad depth measure
        return (-1,-1),(0,0,0),(0,0,0)
    #transform
    (Xp,Yp,Zp)=pointTransform(G,X,Y,Z)
    #project
    (x,y,w)=piD(Xp,Yp,Zp,level)
    ix=int(x+0.5)#good conversion to int
    iy=int(y+0.5)#good conversion to int
    return (ix,iy),(X,Y,Z),(Xp,Yp,Zp)

def warp(Ic,Id,se3,x,y):
    G=se3.matrix()
    return warpMatrix(Ic,Id,G,x,y)

def warpMatrix(I2c,I1d,G):
    '''
    return color in I2c from tranformation from 3d points in I1d and G
    '''
    xMax=I2c.shape[1]
    yMax=I2c.shape[0]
    Iw=np.zeros_like(I2c)-1
    for v in range(yMax):
        #if v%50==0:
        #    print "Warping row ",v
        for u in range(xMax):
            (ix,iy),_,_=warpPoint(I1d,G,u,v)# (u,v) 2D point from I1 
            if inShape(I2c.shape,ix,iy):
                #print "ixiy",ix,iy,u,v
                Iw[iy,ix]=I2c[v,u]
    return Iw

def areColors(d1,c2):
    if d1==0:
        return False
    if c2<0.0:
        return False
    return True

def Ils(I1c,I1d,I2c,G):
    xMax=I1c.shape[1]
    yMax=I1c.shape[0]
    I2w=warpMatrix(I2c,I1d,G)
    #cv2.imshow("I2w",I2w/255.0)
    Ir=np.zeros((yMax,xMax))
    for v in range(yMax):
        #if v%50==0:
        #    print "Error row ",v
        for u in range(xMax):
            c1=np.float32(I1c[v,u])
            c2=np.float32(I2w[v,u])
            d=I1d[v,u]
            if areColors(d,c2):
                cd=c2-c1
                #r=cd*cd
                #print c1,c2,cd,r,d
                Ir[v,u]=cd
            else:
                #print "nan",c1,c2,d
                Ir[v,u]=np.nan
    return Ir
def r(I2w,I1c):
    '''
    return a image of residuals with np.nan on invalid positions.
    '''
    xMax=I1c.shape[1]
    yMax=I1c.shape[0]
    Ir=np.zeros((yMax,xMax))
    for v in range(yMax):
        #if v%50==0:
        #    print "Error row ",v
        for u in range(xMax):
            c1=np.float32(I1c[v,u])
            c2=np.float32(I2w[v,u])
            if c2>=0:
                cd=c2-c1
                Ir[v,u]=cd
            else:#invalid point
                #print "nan",c1,c2,d
                Ir[v,u]=np.nan
    return Ir
   
def Els(Ils):
    error=np.nansum(np.abs(Ils))
    return error/(Ils.shape[0]*Ils.shape[1])

def showRImg(img,name="RImg"):
    xMax=img.shape[1]
    yMax=img.shape[0]
    imax=np.nanmax(img)
    ir=np.float32(np.nan_to_num(img)/imax)
    #print ir.shape,type(ir[0,0]),np.max(ir)
    imgs=cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    for j in range(yMax):
        for i in range(xMax):
            if np.isnan(img[j,i]):
                #print "np.NaN"
                imgs[j,i]=[0,0,1]
    cv2.imshow(name,imgs)

#xi=twist of X the i-th pixel in I1
def Jwp(xi,Xi,G,I1d):
    (ix,iy),(X,Y,Z),(Xp,Yp,Zp)=warpPoint(I1d,G,Xi[0],Xi[1])
    #Intrinsic data for Stum dataset
    level=640/I2c.shape[1]
    fx = 525.0/level  # focal length x
    fy = 525.0/level  # focal length y
    cx = 319.5/level  # optical center x
    cy = 239.5/level  # optical center y
    jw=np.matrix(np.array(2,6))
    jw[0,0]=fx/Zp
    jw[0,1]=0.0
    jw[0,2]=-fx*Xp/Zp**2
    jw[0,3]=-fx*Xp*Yp/Zp**2
    jw[0,4]=fx*(1+Xp**2/Zp**2)
    jw[0,5]=-fx*Yp/Zp
    jw[1,0]=0.0
    jw[1,1]=fy/Xp
    jw[1,2]=-fy*Yp/Zp**2
    jw[1.3]=-fx(1+Yp**2/Zp**2)
    jw[1,4]=fy*Xp*Yp/Zp**2
    jw[1,5]=fy*Xp/Zp
    return jw,(ix,iy)#xi=twist of X the i-th pixel in I1

def Jw((X,Y,Z),l):
    #Intrinsic data for Stum dataset
    level=640/2**l
    fx = 525.0/level  # focal length x
    fy = 525.0/level  # focal length y
    cx = 319.5/level  # optical center x
    cy = 239.5/level  # optical center y
    jw=np.matrix(np.array(2,6))
    jw[0,0]=fx/Z
    jw[0,1]=0.0
    jw[0,2]=-fx*X/Z**2
    jw[0,3]=-fx*X*Y/Z**2
    jw[0,4]=fx*(1+X**2/Z**2)
    jw[0,5]=-fx*Y/Z
    jw[1,0]=0.0
    jw[1,1]=fy/X
    jw[1,2]=-fy*Y/Z**2
    jw[1.3]=-fx(1+Y**2/Z**2)
    jw[1,4]=fy*X*Y/Z**2
    jw[1,5]=fy*X/Z
    return jw
    
def JI(I2c):
    '''
    return two float gray images of gradient of I2c
    '''
    gray = I2c #cv2.cvtColor(I2c, cv2.COLOR_BGR2GRAY)
    I2x=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    I2y=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    return I2x,I2y
    
def Ji(xi,Xi,G,I2x,I2y,I1d):
    jw,(ix,iy)=Jw(xi,Xi,G,I1d)
    dx=I2x[iy,ix]
    dy=I2y[iy,ix]
    ji=np.matrix(np.array([[dx],[dy]]))
    return ji*jw
                        

def J(xi,Xi,G,I2c,I1d):
    xMax=I2c.shape[1]
    yMax=I2c.shape[0]
    j=[]
    I2x,I2y=JI(I2c)
    for v in range(yMax):
        for u in range(xMax):
            ji=Ji(xi,(u,v),G,I2c,I1d)
            j.append(ji)
    return np.matrix(j)
def ComputeScale(r):
    return 1.0
def ComputeWeights(c):
    return 1.0
def ComputeDerivatives(I):
    Ix,Iy=JI(I)
    vIx=np.reshape(Ix,(-1,1))
    vIy=np.reshape(Iy,(-1,1))
    return vIx,vIy

def computeJacobian(Xi,I2xi,I2yi,level):
    '''
    Compute Jacobian of i-th point
    '''
    jw=Jw(Xi,level)
    ji=np.matrix([[I2xi],[I2yi]])
    return ji*jw

def get3DPoints(Id):
    xMax=Id.shape[1]
    yMax=Id.shape[0]
    x=np.zeros((yMax,xMax,3))
    level=640/xMax#pyrLevel
    for v in range(yMax):
        for u in range(xMax):
            X,Y,Z=piI(Id,u,v,level)
            if Z==0:
                x[v,u]=[np.nan,np.nan,np.nan]
            else:
                x[v,u]=[X,Y,Z]
    return np.array(x)

def makeFlat(X,I2x,I2y,I2w):
    xMax=I2w.shape[1]
    yMax=I2w.shape[0]
    level=640/xMax#pyrLevel
    fX=[]
    fI2x=[]
    fI2y=[]
    for v in range(yMax):
        for u in range(xMax):
            if not np.isnan(I2w[v,u]):
                fX  .append(X[v,u])
                fI2x.append(I2x[v,u])
                fI2y.append(I2y[v,u])
    return np.matrix(fX),np.matrix(fI2x),np.matrix(fI2y)

def computeJacobianAndError(I1,D1,I2,xi,level):
    #Work in 2D
    I2w=warpMatrix(I2,D1,xi.getMatrix())#I2[warp(x,xi)]
    r=r(I2w,I1)#r=I2w-I1
    sigma=ComputeScale(r)
    W=ComputeWeights(r/sigma)
    I2x,I2y=ComputeDerivatives(I2w)
    X=get3DPoints(D1)
    #work in 1D
    fX,fI2x,fI2y=makeFlat(X,I2x,I2y,I2w)
    print fX.shape,fI2x.shape,fI2x.shape
    n=fX.shape[0]
    for i in range(n):
        J[i]=computeJacobian(fX[i],fI2x[i],fI2y[i],level)
    return J,W,r
    
def estimateCameraMotion(I1pyr,D1pyr,I2pyr,xiInitial):
    epsilon=0.001
    kMax=10
    xi=xiInitial
    dXi=se3(0)
    for i in range(3,-1,-1):
        e=0
        eLast=np.Infinity
        k=0
        while (eLast-e)>epsilon and k<kMax:
            J,W,r=computeJacobianAndError(I1pyr[i],D1pyr[i],I2pyr[i],xi+dXi,level=i)
            eLast=e
            n=J.shape[0]
            e=r.T*W*r/n
            if e>eLast:
                dXi=se3(0)
            else:
                xi=xi+dXi
                dXi=la.inv(J.T*W*J)*J.T*W*r
            k+=1
    return xi 

dw=math.pi/(360.0*5)
dv=0.001
def Grad(Els0,I1c,I1d,I2c,Ga):
    Sgn=[1,1,1,1,1,1]
    #Rotations1
    #dw=math.pi/(360.0*2)
    GRx=getRotMatX(dw)
    iRx=Ils(I1c, I1d, I2c, Ga*GRx)
    ElsRx=Els(iRx)
    showRImg(iRx, "IlsRx")
    dRx=Els0-ElsRx
    print "Rx",ElsRx,dRx
    GRy=getRotMatY(dw)
    iRy=Ils(I1c, I1d, I2c, Ga*GRy)
    ElsRy=Els(iRy)
    showRImg(iRy, "IlsRy")
    dRy=Els0-ElsRy
    print "Ry",ElsRy,dRy
    GRz=getRotMatX(dw)
    iRz=Ils(I1c, I1d, I2c, Ga*GRz)
    ElsRz=Els(iRz)
    showRImg(iRz, "IlsRz")
    dRz=Els0-ElsRz
    print "Rz",ElsRz,dRz
    #Translation
    #dv=0.001
    GTx=np.matrix(np.eye(4,4))
    GTx[0,3]=dv
    iTx=Ils(I1c,I1d,I2c,Ga*GTx)
    ElsTx=Els(iTx)
    showRImg(iTx, "IlsTx")
    dTx=Els0-ElsTx
    print "Tx",ElsTx,dTx
    GTy=np.matrix(np.eye(4,4))
    GTy[1,3]=dv
    iTy=Ils(I1c,I1d,I2c,Ga*GTy)
    ElsTy=Els(iTy)
    showRImg(iTy, "IlsTy")
    dTy=Els0-ElsTy
    print "Ty",ElsTy,dTy
    GTz=np.matrix(np.eye(4,4))
    GTz[2,3]=dv
    iTz=Ils(I1c,I1d,I2c,Ga*GTz)
    ElsTz=Els(iTz)
    showRImg(iTz, "IlsTz")
    dTz=Els0-ElsTz
    print "Tz",ElsTz,dTz
      
#     Ga1=GTz*GTy*GTx*GRx*GRy*GRz*Ga
#     print Ga1
#     iGa1=Ils(I1c,I1d,I2c,Ga1)
#     ElsGa1=Els(iGa1)
#     showRImg(iGa1, "IlsA1")
#     dGa1=Els0-ElsGa1
#     print "Ga",ElsGa1,dGa1
    return dRx,dRy,dRz,dTx,dTy,dTz

def updateG(Ga,Sgn):
    print Sgn
    #Rotations1
    #dw=math.pi/(360.0*2)
    GRx=getRotMatX(Sgn[0]*dw)
    GRy=getRotMatY(Sgn[1]*dw)
    GRz=getRotMatX(Sgn[2]*dw)
    #Translation
    #dv=0.001
    GTx=np.matrix(np.eye(4,4))
    GTx[0,3]=Sgn[3]*dv
    GTy=np.matrix(np.eye(4,4))
    GTy[1,3]=Sgn[4]*dv
    GTz=np.matrix(np.eye(4,4))
    GTz[2,3]=Sgn[5]*dv
    Ga1=GTz*GTy*GTx*GRx*GRy*GRz*Ga
    print Ga1
    iGa1=Ils(I1c,I1d,I2c,Ga1)
    ElsGa1=Els(iGa1)
    showRImg(iGa1, "IlsA1")
    dGa1=Els0-ElsGa1
    print "Ga",ElsGa1,dGa1
    return Ga1
    
ptsTf,clrTf,ptsT,clrT,imgT,depT=getPointCloudFromRangeAsso(assoFn,0,1)
ptsMf,clrMf,ptsM,clrM,imgM,depM=getPointCloudFromRangeAsso(assoFn,1,1)

#Color to Gray float32
imgTg=np.float32(cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY))/255.0
imgMg=np.float32(cv2.cvtColor(imgM, cv2.COLOR_BGR2GRAY))/255.0

#print imgTg.shape,type(imgTg[0,0]),np.max(imgTg)

p1=getPyr(imgTg)
pd=getPyr(depT)
p2=getPyr(imgMg)

level=1
I1c=p1[level]
I1d=pd[level]
I2c=p2[level]
cv2.imshow("im1",I1c)
cv2.imshow("im2",I2c)
G=np.eye(4,4)
#I2w=warpMatrix(I2c,I1d,G)
#cv2.imshow("I2w",I2w/255.0)
i=Ils(I1c,I1d,I2c,G)
showRImg(i, "Ils")
Els0=Els(i)
print Els0
for i in range(10):
    g=Grad(Els0, I1c, I1d, I2c, G)
    s=np.sign(g)
    G=updateG(G,s)

while True:
    #rate(10)      
    k = cv2.waitKey(10)
    if not k == -1:
        break


if __name__ == '__main__':
    pass
'''
Created on Aug 25, 2015

@author: Francisco Dominguez
'''
import numpy as np
from numpy import random
from visual import points
import cv2
import qmath
 
# q=qmath.quaternion('1+2i+3j-1k')
# print q
# p=q.unitary()
# print qmath.imag(q)

print range(3,-1,-1)

img_size=10
apts = []
ref_distrs = []
mean = (0.1 + 0.8*random.rand(3)) * img_size
a = (random.rand(3, 3)-0.5)*img_size*0.1
cov = np.dot(a.T, a) + img_size*0.05*np.eye(3)
n = 100 + random.randint(900)
pts = random.multivariate_normal(mean, cov, n)
apts.append( pts )
ref_distrs.append( (mean, cov) )
nppoints = np.float32( np.vstack(apts) )
#points(pos=nppoints)


puntos=[[10,0,0],
        [-10,-1,0],
        [5,0,-5],
        [-5,0,5],
        [2,2,2],
        [-2,2,-2]]
npts=np.float32(puntos)/len(puntos)
#npts=np.float32(nppoints)/nppoints.shape[0]
cov=npts.T.dot(npts)
print cov
w, u, vt = cv2.SVDecomp(cov)
print "w",w
print "u",u
print "vt",vt
l=np.sqrt(w)
ro=l[2]/np.sum(l)
print ro
points(pos=npts)
if __name__ == '__main__':
    pass
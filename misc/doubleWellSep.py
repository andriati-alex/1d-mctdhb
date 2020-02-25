import sys
import numpy as np
import scipy.optimize as opt
from math import log, sqrt

def findparams(x,h,pos):
    b = x[0]
    s = x[1]
    pos2 = pos**2
    comp = np.zeros(2)
    comp[0] = log(b**2/s**2) - pos2 / (2*s**2)
    comp[1] = (b**2  - s**2) - pos2 / 2 - h
    return comp

height = float(sys.argv[1])
xmin = float(sys.argv[2])
sol = opt.root(findparams,[2.7856,0.74565],args=(height,xmin));
Aquartic = sqrt(4*height/xmin**2)
Bquartic = Aquartic / xmin
print("\nWith Gaussian use :  1.0  %.15lf  %.15lf" % (sol.x[0],sol.x[1]))
print("\nWith Quartic use  :  %.15lf  %.15lf\n" % (Aquartic,Bquartic))

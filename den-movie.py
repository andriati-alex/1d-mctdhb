import sys;
import numpy as np;
import matplotlib.pyplot as plt;
import mcanalysis as mc;

from pathlib import Path;
from matplotlib.animation import FuncAnimation;

fname_prefix = sys.argv[1];
folder = str(Path.home()) + '/programs/1d-mctdhb/output/';
full_path = folder + fname_prefix;

S = np.loadtxt(full_path + '_orb_realtime.dat',dtype=np.complex128);
Sx = np.loadtxt(full_path + '_orb_realtime.dat',dtype=np.complex128);
rho = np.loadtxt(full_path + '_rho_realtime.dat',dtype=np.complex128);
rhox = np.loadtxt(full_path + '_rho_realtime.dat',dtype=np.complex128);
conf = np.loadtxt(full_path + '_conf.dat');
domain = np.loadtxt(full_path + '_domain_realtime.dat');

x = np.linspace(domain[0,1],domain[0,2],int(domain[0,3]))
xi = domain[:,4]
xf = domain[:,5]
t = domain[:,0]

Npar = int(conf[0])
Norb = int(conf[1])

Nsteps = t.size;

# tocc = mc.TimeOccupation(4,3,rho)
den = mc.TimeDensity(Norb,x.size,rho,S)
denx = mc.TimeDensity(Norb,x.size,rhox,Sx)

animTime = 50000 # in miliseconds

fig = plt.figure(figsize=(10,10));
ax = plt.gca();

line, = ax.plot([],[],'-');
lineX, = ax.plot([],[],'r--');

# initialization function: plot the background of each frame
def init():
    line.set_data([], []);
    lineX.set_data([], []);
    ax.set_xlim(xi[0],xf[0]);
    ax.set_ylim(0,den.max());
    return [line,lineX]

# animation function. This is called sequentially
def animate(i):
    line.set_data(x,den[i]);
    lineX.set_data(x,denx[i]);
    return [line,lineX]

anim = FuncAnimation(fig, animate, init_func=init,
                     frames=Nsteps, interval=animTime/Nsteps, blit=True);

plt.show();

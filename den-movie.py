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
rho = np.loadtxt(full_path + '_rho_realtime.dat',dtype=np.complex128);
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

animTime = 50000 # in miliseconds

fig = plt.figure(figsize=(8,7));
ax = plt.gca();
time_template = 't = {:.3f}'

line, = ax.plot([],[],'-');
timetext = ax.text(0.05,0.95,'',transform=ax.transAxes,ha='left',va='top')

# initialization function: plot the background of each frame
def init():
    line.set_data([], []);
    timetext.set_text('')
    ax.set_xlim(xi[0],xf[0]);
    ax.set_ylim(den.min()*0.8,den.max()*1.1);
    return line, timetext

# animation function. This is called sequentially
def animate(i):
    ax.set_xlim(xi[i],xf[i])
    line.set_data(x,den[i])
    timetext.set_text(time_template.format(t[i]))
    if (xf[i] > xf[i-1]): plt.draw()
    return line, timetext

anim = FuncAnimation(fig, animate, init_func=init,
                     frames=Nsteps, interval=animTime/Nsteps, blit=True);

plt.show();

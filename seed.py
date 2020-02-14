
"""
=============================================================================

    SCRIPT TO GENERATE INITIAL CONDITION DATA FOR MCTDHB

    Generate 4 files in setup/ folder:

    1) 'fileId_conf.dat' contains the configurational parameters number
        of particle and number of orbitals. It also contains the domain
        grid information [xi,xf] with 'Ndiv'steps between  the  points.
        Finally it also have the time step dt and  number  of  steps to
        propagate in time

        Npar Norb Ndiv xi xf dt Nsteps

    2) 'fileId_orb.dat' contains:

        Matrix whose each column is an orbital and the values of
        the orbitals in discretized positions is given along the
        rows. For example if Norb = 3 then:

                Orbital #1     Orbital #2     Orbital #3

           x1 |   f1(x1)         f2(x1)         f3(x1)
           x2 |   f1(x2)         f2(x2)         f3(x2)
              |     .              .              .
              |     .              .              .
              |     .              .              .
           xm |   f1(xm)         f2(xm)         f3(xm)

    3) 'fileId_coef.dat' contains a column vector of size:

         (Npar + Norb - 1)!   | with values of coefficietns of
        --------------------  | all possible configurations on
        (Npar)!  (Norb - 1)!  | occupations in a Fock state

    4) 'fileId_eq.dat' file with equation parameters, that is filled
        with arbitrary values and  one must change as wish. They are
        given in the following order: 'a2'  as the  coefficient that
        multiply 2nd order derivative, 'a1' the imag. part of coeff.
        that goes with 1st order derivative, 'g' the interaction and
        'pk' a set of parameters for the one-body potential. See the
        linearPotential source file.

        a2 a1 g p1 p2 p3

=============================================================================
"""

import sys;
import numpy as np;
import scipy.special as ss;

from math import pi;
from math import sqrt;
from scipy.integrate import simps;
from numba import jit, prange, int32, uint32, uint64;

lf = np.float128;
lc = np.complex256;



@jit(uint64(uint64), nopython=True, nogil=True)
def fac(n):
    """ return n! """
    nfac = 1;
    for i in prange(2, n + 1): nfac = nfac * i;
    return nfac;



@jit(uint32(uint32,uint32), nopython=True, nogil=True)
def NC(Npar, Norb):
    """ return (Npar + Norb - 1)! / ( (Npar)! x (Norb - 1)! )"""
    n = 1
    j = 2
    if (Norb > Npar) :
        for i in prange(Npar + Norb - 1, Norb - 1, -1) :
            n = n * i
            if (n % j == 0 and j <= Npar) :
                n = n / j
                j = j + 1
        for k in prange(j,Npar+1) : n = n / k
        return n
    else :
        for i in prange(Npar + Norb - 1, Npar, -1) :
            n = n * i;
            if (n % j == 0 and j <= Norb - 1) :
                n = n / j
                j = j + 1
        for k in prange(j,Norb) : n = n / k
        return n



@jit((int32, int32, int32, int32[:]), nopython=True, nogil=True)
def IndexToFock(k,N,M,v):
    """
    k : Index of configuration-state coefficient
    N : # of particles
    M : # of orbitals
    v : End up with occupation vector(Fock state) of length Morb
    """
    x = 0
    m = M - 1
    for i in prange(0,M,1) : v[i] = 0
    while (k > 0) :
        while (k - NC(N,m) < 0) : m = m - 1
        k = k - NC(N, m)
        v[m] = v[m] + 1
        N = N - 1
    if (N > 0) : v[0] = v[0] + N



def renormalize(f,dx): return f / sqrt(simps(abs(f)**2,dx=dx));



def HarmonicTrap(Norb, x, S, omega):
    phase = (np.random.random(Norb) - 0.5) * 2 * pi;
    for n in range(Norb):
        her = ss.eval_hermite(n, sqrt(omega) * x);
        div = pow((2 ** n) * fac(n), 0.5) * pow(pi / omega, 0.25);
        S[n,:] = np.exp(- omega * x * x / 2 + 1.0j * phase[n]) * her / div;



def RingBarrier(Norb,x,S,L,k):
    """
    Orthonormal set of functions that are periodic and vanish at the centre
    """
    A = sqrt(2.0/L) * np.exp(2.0j*np.pi*k*x/L)
    for i in range(0, Norb):
        S[i,:] = A * np.sin(2*pi*(i+1)*x/L)



def Ring(Norb,x,S,L):
    """ Plane waves with periodic boundary conditions """
    S[0,:] = 1.0 / sqrt(L)
    Kminus = 0
    Kplus = 0
    for i in range(1, Norb) :
        if (i % 2 == 0) :
            Kminus = Kminus + 1
            S[i,:] = np.exp(-2.0j*pi*Kminus*x/L) / sqrt(L)
        else :
            Kplus = Kplus + 1
            S[i,:] = np.exp(+2.0j*pi*Kplus*x/L) / sqrt(L)



def ThermalCoef(Npar,Norb):
    """
    Consider the energy proportional to the square  of  orbital  number
    and them consider a thermal-like distribution  with  the fock-state
    coefficient decreasing exponentially according  to  the  number  of
    occupation in each orbital(j) times the square of orbital number(j)
    (representing the energy)
    -------------------------------------------------------------------
    C = array of coefficient of size of # of all possible fock config.
    Npar = # of particles
    Norb = # of orbitals
    """
    nc = NC(Npar,Norb)
    C = np.empty(nc,dtype=lc)
    # Fock vector for each coefficient
    v = np.empty(Norb,dtype=np.int32)
    phase = np.exp(2 * pi * 1.0j * (np.random.random(nc) - 0.5))

    beta = 2.0

    for l in range(nc) :
        IndexToFock(l,Npar,Norb,v)
        decay = 0
        # Sum total "energy" of the configuration into "thermal" decay
        for j in range(Norb) : decay = decay - beta * float(v[j] * j) / Npar
        # put the random phase with "thermal" exponential decay
        C[l] = phase[l] * np.exp(decay,dtype=lf)
    # renormalize coefficients
    return C / sqrt((abs(C)**2).sum())










########################################################################
###################                                 ####################
###################        SCRIPT STARTS HERE       ####################
###################   read command line arguments   ####################
###################                                 ####################
########################################################################

Npar  = int(sys.argv[1])   # Number of Particles
Norb  = int(sys.argv[2])   # Number of orbitals
xi    = float(sys.argv[3]) # initial position
xf    = float(sys.argv[4]) # final position
Ndiv  = int(sys.argv[5])   # Number of division in the grid domain
fname = sys.argv[6]        # function Identification - one of the above

# extra parameters to the seed function
params = []
# read them if there was passed any
for i in range(7, len(sys.argv)): params.append(lf(sys.argv[i]))
params = tuple(params)





########################################################################
############                                               #############
############ Call chosen routine to generate initial data  #############
############       within the specified grid domain        #############
############                                               #############
########################################################################

x  = np.linspace(xi,xf,Ndiv+1)
L  = xf - xi
dx = (xf-xi) / Ndiv

Orb = np.zeros([Norb,x.size],dtype=lc);  # orbitals
C = ThermalCoef(Npar, Norb) # setup coefficients

# Setup orbitals according to the Identification passed
if (fname == 'HarmonicTrap') :
    Id_name = 'harmonicTrap-' + str(Npar) + '-' + str(Norb)
    HarmonicTrap(Norb,x,Orb,params[0])
elif (fname == 'Ring' or fname == 'ring') :
    Id_name = 'ring-' + str(Npar) + '-' + str(Norb)
    Ring(Norb,x,Orb,L);
elif (fname == 'RingBarrier') :
    Id_name = 'ringBarrier-' + str(Npar) + '-' + str(Norb)
    RingBarrier(Norb,x,Orb,L,params[0])
else : raise IOError('\n\nSeed function name not implemented.\n\n')





###################################################################
#################                               ###################
#################      RECORD INITIAL DATA      ###################
#################                               ###################
###################################################################

folder = './input/'

np.savetxt(folder+Id_name+'_orb.dat',Orb.T,fmt='%.15E')
np.savetxt(folder+Id_name+'_coef.dat',C.T,fmt='%.15E')

# The two last values are time step and the amount of steps to propagete
# the initial condition that are just suggestion  and must be adapted to
# the specific problem
f = open(folder + Id_name + '_conf.dat', 'w')
f.write( '%d %d %d %.5f %.5f 0.001 10000' % (Npar,Norb,Ndiv,xi,xf))
f.close()

# Arbitrary values for equation parameters. The value must be change.
f = open(folder + Id_name + '_eq.dat','w')
f.write( '-0.5 0.0 1.0 1.0 0.0 0.0')
f.close()


import sys;
import numpy as np;
import MCTDHBmodule as mc;
import scipy.special as ss;

from scipy.integrate import simps;
from math import pi;
from math import sqrt;
from math import factorial as fac;
from pathlib import Path;





"""
=============================================================================


    SCRIPT TO GENERATE INITIAL CONDITION DATA FOR MCTDHB
    ----------------------------------------------------

    Generate 3 files in setup/ folder:



    1) 'MC_fileId_config.dat' contains the numbers:

        Npar Morb Mdiv xi xf



    2) 'MC_fileId_orb.dat' contains:

        Matrix whose each column is an orbital and the values of
        the orbitals in discretized positions is given over  the
        rows. For example if Morb = 3 then:

                Orbital #1     Orbital #2     Orbital #3

           x1 |   f1(x1)         f2(x1)         f3(x1)
           x2 |   f1(x2)         f2(x2)         f3(x2)
              |     .              .              .
              |     .              .              .
              |     .              .              .
           xm |   f1(xm)         f2(xm)         f3(xm)



    3) 'MC_fileId_coef.dat' contains a column vector of size:

         (Npar + Morb - 1)!   | with values of coefficietns of
        --------------------  | all possible configurations on
        (Npar)!  (Morb - 1)!  | occupations in a Fock state


=============================================================================
"""





lf = np.float128;
lc = np.complex256;





def renormalize(f, dx): return f / sqrt( simps(abs(f)**2, dx = dx) );





def HarmonicTrap(Morb, x, S, omega):
    phase = (np.random.random(Morb) - 0.5) * 2 * pi;
    for n in range(Morb):
        her = ss.eval_hermite(n, sqrt(omega) * x);
        div = pow((2 ** n) * fac(n), 0.5) * pow(pi / omega, 0.25);
        S[n,:] = np.exp(- omega * x * x / 2 + 1.0j * phase[n]) * her / div;





def RingBarrier(Morb, x, S, omega):
    i = 0;
    if (int(10 * omega) % 10 == 0):
        S[0,:] = np.exp(0.5j * pi * omega * x) * np.sin(pi * omega * x * 0.5);
        i = 1;
    m = 1;
    for k in range(0, Morb - i):
        if (m == omega) : m = m + 1;
        S[k + i,:] = np.exp(1.0j * np.pi * omega * x) * np.sin(m * np.pi * x)
        m = m + 1;





def Ring(Morb, x, S):
    """
    Setup matrix S with eigenstates of  angular  momentum  in ascending
    if the # of orbitals(Morb) is odd. Odd orbitals have minus signs of
    angular momentum while even orbitals have plus sign.
    -------------------------------------------------------------------

    x    = array of discretized position domain
    Morb = # of orbitals
    """

    S[0,:] = 1.0 / sqrt(2 * pi);
    for i in range(2, Morb, 2):
        S[i - 1,:] = np.exp(- 1.0j * (i / 2) * x, dtype=lc) / sqrt(2 * pi);
        S[i, :]    = np.exp(  1.0j * (i / 2) * x, dtype=lc) / sqrt(2 * pi);





def ThermalCoef(Npar, Morb, C):
    """
    Consider the energy proportional to the square  of  orbital  number
    and them consider a thermal-like distribution  with  the fock-state
    coefficient decreasing exponentially according  to  the  number  of
    occupation in each orbital(j) times the square of orbital number(j)
    (representing the energy)
    -------------------------------------------------------------------

    C = array of coefficient of size of # of all possible fock config.
    Npar = # of particles
    Morb = # of orbitals
    beta = decay rate (something like ~ 1 / temperature)
    """
    nc = mc.NC(Npar, Morb);
    v = np.empty(Morb, dtype=np.int32); # Fock vector for each coefficient
    phase = np.exp(2 * pi * 1.0j * (np.random.random(nc) - 0.5));

    beta = 2.0;

    for l in range(nc):
        mc.IndexToFock(l, Npar, Morb, v);
        prod = 1.0;
        for j in range(Morb):
            decay = - beta * float(v[j] * j) / Npar;
            prod  = prod * np.exp( decay , dtype=lf );
        # put the random phase with 'thermal' exponential decay
        C[l] = phase[l] * prod;

    # renormalize coefficients
    Norm = 0;
    for l in range(nc): Norm = Norm + abs(C[l]) * abs(C[l]);
    for l in range(nc): C[l] = C[l] / sqrt(Norm);










###################      Read command line options      ####################

Npar  = int(sys.argv[1]);   # Number of Particles
Morb  = int(sys.argv[2]);   # Number of orbitals
xi    = float(sys.argv[3]); # initial position
xf    = float(sys.argv[4]); # final position
Mdiv  = int(sys.argv[5]);   # Number of divisions slices in domain
fname = sys.argv[6];   # function to generate Id

params = []; # extra parameters to pass to the function
for i in range(7, len(sys.argv)): params.append( lf(sys.argv[i]) );
params = tuple(params);





############### Call subroutines to setup initial condition ################

x  = np.linspace(xi, xf, Mdiv + 1);
dx = (xf - xi) / Mdiv;

Orb = np.zeros([Morb, x.size], dtype=lc);  # orbitals

C = np.zeros(mc.NC(Npar, Morb), dtype=lc); # coeficients

ThermalCoef(Npar, Morb, C);

if   (fname == 'HarmonicTrap') :

    Id_name = 'MCHarmonicTrap-' + str(Npar) + '-' + str(Morb);
    HarmonicTrap(Morb, x, Orb, params[0]);

elif (fname == 'Ring') :

    Id_name = 'MCRing-' + str(Npar) + '-' + str(Morb);
    Ring(Morb, x, Orb);

elif (fname == 'RingBarrier') :

    Id_name = 'MCRingBarrier-' + str(Npar) + '-' + str(Morb);
    RingBarrier(Morb, x, Orb, params[0]);

else : raise IOError('\n\nInitial function name not implemented.\n\n');





#################              Record Data              ###################

folder = str(Path.home()) + '/AndriatiLibrary/1d-mctdhb/input/';

np.savetxt(folder + Id_name + '_orb.dat', Orb.T, fmt='%.15E');
np.savetxt(folder + Id_name + '_coef.dat', C.T, fmt='%.15E');

f = open(folder + Id_name + '_conf.dat', 'w');
f.write( '%d %d %d %.15f %.15f' % (Npar, Morb, Mdiv, xi, xf) );
f.close();

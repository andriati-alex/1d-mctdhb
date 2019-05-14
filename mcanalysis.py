
import cmath;
import numpy as np;
import scipy.linalg as la;

from math import sqrt, pi;
from scipy.integrate import simps;
from numba import jit, prange, int32, uint32, uint64, int64, float64, complex128;





"""
=============================================================================


    MODULE DIRECTED TO COLLECT OBSERVABLES FROM MCTDHB
    --------------------------------------------------


    The functions contained in this module support the analysis of results 
    from (imaginary/real)time propagation. Some of the functions  are  the
    same as those used in C language,  to  reduce  the  amount of data set
    needed. Numba package usage may cause delay in importation.


=============================================================================
"""





def derivative(dx, f):
    """
    CALLING :
    -------
    (1D array of size of f) = derivative(dx, f)

    comments :
    --------
    Use finite differences to compute derivative with 4-th order error
    """
    n = f.size;
    dfdx = np.zeros(n, dtype=np.complex128);

    dfdx[0] = ( f[n-3] - f[2] + 8 * (f[1] - f[n-2]) ) / (12 * dx);
    dfdx[1] = ( f[n-2] - f[3] + 8 * (f[2] - f[0]) ) / (12 * dx);
    dfdx[n-2] = ( f[n-4] - f[1] + 8 * (f[0] - f[n-3]) ) / (12 * dx);
    dfdx[n-1] = dfdx[0]; # assume last point as the boundary

    for i in range(2, n - 2):
        dfdx[i] = ( f[i-2] - f[i+2] + 8 * (f[i+1] - f[i-1]) ) / (12 * dx);
    return dfdx;




def centered_derivative(dx, f):
    n = f.size;
    dfdx = np.zeros(n,dtype=np.complex128)

    dfdx[0] = (f[1] - f[n-2]) / (2 * dx);
    dfdx[n-1] = dfdx[0];

    for i in range(1,n-1): dfdx[i] = (f[i+1] - f[i-1]) / (2 * dx);

    return dfdx;





def dxFFT(dx, f):
    """
    CALLING :
    -------
    (1D array of size of f) = derivative(dx, f)

    comments :
    --------
    Use Fourier-Transforms to compute derivative
    """
    n = f.size - 1;
    k = 2 * pi * np.fft.fftfreq(n, dx);
    dfdx = np.zeros(f.size, dtype = np.complex128);
    dfdx[:n] = np.fft.fft(f[:n], norm = 'ortho');
    dfdx[:n] = np.fft.ifft(1.0j * k * dfdx[:n], norm = 'ortho');
    dfdx[n] = dfdx[0];
    return dfdx;





def L2normFFT(dx, func, norm = 1):
    """
    Normalized according to L2 norm in the momentum space.

    CALLING
    -------
    freq_vector , fft_of_func = L2normFFT(dx,func,norm);
    """

    n = func.size;

    k = 2 * pi * np.fft.fftfreq(n,dx);
    dk = k[1] - k[0];

    fftfunc = np.fft.fft(func);

    if (n % 2 == 0) : j = int(n / 2) - 1;
    else            : j = int((n - 1) / 2);

    k = np.concatenate( [ k[j+1:] , k[:j+1] ] );
    fftfunc = np.concatenate( [ fftfunc[j+1:] , fftfunc[:j+1] ] );

    return k, fftfunc * norm / np.sqrt(simps(abs(fftfunc)**2,dx=dk));





def DnormFFT(dx, func, norm = 1):
    """
    Normalized Fourier 'Series' for periodic functions with
    non-vanishing boundaries.

    CALLING
    -------
    freq_vector , fft_of_func = L2normFFT(dx,func,norm);
    """

    n = func.size;

    k = 2 * pi * np.fft.fftfreq(n,dx);

    fftfunc = np.fft.fft(func,norm='ortho');

    if (n % 2 == 0) : j = int(n / 2) - 1;
    else            : j = int((n - 1) / 2);

    k = np.concatenate( [ k[j+1:] , k[:j+1] ] );
    fftfunc = np.concatenate( [ fftfunc[j+1:] , fftfunc[:j+1] ] );

    return k, fftfunc / sqrt((abs(fftfunc)**2).sum());





@jit([ uint32(uint32) , uint64(uint64) ], nopython=True, nogil=True)

def fac(n):
    """ return n! """
    nfac = 1;
    for i in prange(2, n + 1): nfac = nfac * i;
    return nfac;





@jit( uint32(uint32, uint32), nopython=True, nogil=True)

def NC(Npar, Morb):
    """ return (Npar + Morb - 1)! / ( (Npar)! x (Morb - 1)! )"""
    n = 1;
    if (Npar > Morb):
        for i in prange(Npar + Morb - 1, Npar, -1): n = n * i;
        return n / fac(Morb - 1);
    else :
        for i in prange(Npar + Morb - 1, Morb - 1, -1): n = n * i;
        return n / fac(Npar);





@jit( [ (int32, int32, int32, int32[:]), (int64, int32, int32, int32[:]) ],
      nopython=True, nogil=True)

def IndexToFock(k, N, M, v):
    """
    Calling: (void) IndexToFock(k, N, M, v)
    -------

    Arguments:
    ---------
    k : Index of configuration-state coefficient
    N : # of particles
    M : # of orbitals
    v : End up with occupation vector(Fock state) of length Morb
    """
    x = 0;
    m = M - 1;
    for i in prange(0, M, 1): v[i] = 0;
    while (k > 0):
        while (k - NC(N, m) < 0): m = m - 1;
        x = k - NC(N, m);
        while (x >= 0):
            v[m] = v[m] + 1;
            N = N - 1;
            k = x;
            x = x - NC(N, m);
    for i in prange(N, 0, -1): v[0] = v[0] + 1;





@jit( int32(int32, int32, int32[:,:], int32[:]) , nopython=True, nogil=True)

def FockToIndex(N, M, NCmat, v):
    """
    Calling: (int) = FockToIndex(N, M, NCmat, v)
    --------
    k = Index of Fock Configuration Coeficient of v

    arguments:
    ----------
    N     : # of particles
    M     : # of orbitals
    NCmat : see GetNCmat function
    """
    n = 0;
    k = 0;
    for i in prange(M - 1, 0, -1):
        n = v[i]; # Number of particles in the orbital
        while (n > 0):
            k = k + NCmat[N][i]; # number of combinations needed
            N = N - 1;           # decrease the number of particles
            n = n - 1;
    return k;





@jit( [ (int32, int32, int32[:,:]), (int32, int32, int64[:,:]) ],
      nopython=True, nogil=True)

def MountNCmat(N, M, NCmat):
    """ Auxiliar of GetNCmat """
    for i in prange(0, N + 1, 1):
        NCmat[i][0] = 0;
        for j in prange(1, M + 1, 1): NCmat[i][j] = NC(i, j);





@jit( (int32, int32, int32[:,:]) , nopython=True, nogil=True)

def MountFocks(N, M, IF):
    """ Auxiliar of GetFocks """
    for k in prange(0, NC(N, M), 1): IndexToFock(k, N, M, IF[k]);





def GetNCmat(N, M):
    """
    Calling: (numpy 2D array of ints) = GetNCmat(N, M)
    --------
    Returned matrix NC(n,m) = (n + m - 1)! / ( n! (m - 1)! ).

    arguments:
    ----------
    N: # of particles
    M: # of orbitals
    """
    NCmat = np.empty([N + 1, M + 1], dtype=np.int32);
    MountNCmat(N, M, NCmat);
    return NCmat;





def GetFocks(N, M):
    """
    Calling : (numpy 2D array of ints) = GetFocks(N, M)
    -------
    Row k has the occupation vector corresponding to C[k].

    arguments :
    ---------
    N : # of particles
    M : # of orbitals
    """
    IF = np.empty([NC(N, M), M], dtype=np.int32);
    MountFocks(N, M, IF);
    return IF;





@jit( (int32, int32, int32[:,:], int32[:,:], complex128[:], complex128[:,:]),
      nopython=True, nogil=True)

def OBrho(N, M, NCmat, IF, C, rho):

    """
    Calling : (void) OBrho(N, M, NCmat, IF, C, rho)
    -------
    Setup rho argument with one-body densit matrix

    arguments :
    ---------
    N     : # of particles
    M     : # of orbitals (also dimension of rho)
    NCmat : see function GetNCmat
    IF    : see function GetFocks
    C     : coeficients of Fock-configuration states
    rho   : Empty M x M matrix. End up configured with values
    """

    # Initialize variables
    j    = 0;
    mod2 = 0.0;
    nc   = NCmat[N][M];
    RHO  = 0.0 + 0.0j;

    for k in prange(0, M, 1):

        RHO = 0.0 + 0.0j;

        for i in prange(0, nc, 1):
            mod2 = C[i].real * C[i].real + C[i].imag * C[i].imag;
            RHO  = RHO + mod2 * IF[i][k];

        rho[k][k] = RHO;

        for l in prange(k + 1, M, 1):

            RHO = 0;

            for i in prange(0, nc, 1):
                if (IF[i][k] < 1): continue;
                IF[i][k] -= 1;
                IF[i][l] += 1;
                j = FockToIndex(N, M, NCmat, IF[i]);
                IF[i][k] += 1;
                IF[i][l] -= 1;
                RHO = RHO + C[i].conjugate() * C[j] * \
                      sqrt((IF[i][l] + 1) * IF[i][k]);

            rho[k][l] = RHO;
            rho[l][k] = RHO.conjugate();

# End OBrho routine ------------------------------------------------------










@jit( (int32, int32, int32[:,:], int32[:,:], complex128[:], complex128[:]),
      nopython=True, nogil=True)

def TBrho(N, M, NCmat, IF, C, rho):

    """
    Calling : (void) OBrho(N, M, NCmat, IF, C, rho)
    -------
    Setup rho argument with two-body density matrix

    arguments :
    ---------
    N     : # of particles
    M     : # of orbitals (also dimension of rho)
    NCmat : see function GetNCmat
    IF    : see function GetFocks
    C     : coeficients of Fock-configuration states
    rho   : Empty M x M matrix. End up configured with values
    """

    # index result of conversion FockToIndex
    j = int(0);

    # occupation vector
    v = np.empty(M, dtype=np.int32);
    
    mod2 = float(0);   # |Cj| ^ 2
    sqrtOf = float(0); # Factors from the action of creation/annihilation

    RHO = float(0);

    # Auxiliar to memory access of two-body matrix
    M2 = M * M;
    M3 = M * M * M;

    nc = NCmat[N][M];





    # ---------------------------------------------
    # Rule 1: Creation on k k / Annihilation on k k
    # ---------------------------------------------------------------------
    for k in prange(0, M, 1):

        RHO = 0;

        for i in prange(0, nc, 1):
            mod2 = C[i].real * C[i].real + C[i].imag * C[i].imag;
            RHO  = RHO + mod2 * IF[i][k] * (IF[i][k] - 1);

        rho[k + M * k + M2 * k + M3 * k] = RHO;
    # ---------------------------------------------------------------------



    # ---------------------------------------------
    # Rule 2: Creation on k s / Annihilation on k s
    # ---------------------------------------------------------------------
    for k in prange(0, M, 1):

        for s in prange(k + 1, M, 1):

            RHO = 0;

            for i in prange(0, nc, 1):
                mod2 = C[i].real * C[i].real + C[i].imag * C[i].imag;
                RHO += mod2 * IF[i][k] * IF[i][s];

            # commutation of bosonic operators is used
            # to fill elements by exchange  of indexes
            rho[k + s * M + k * M2 + s * M3] = RHO;
            rho[s + k * M + k * M2 + s * M3] = RHO;
            rho[s + k * M + s * M2 + k * M3] = RHO;
            rho[k + s * M + s * M2 + k * M3] = RHO;
    # ---------------------------------------------------------------------



    # ---------------------------------------------
    # Rule 3: Creation on k k / Annihilation on q q
    # ---------------------------------------------------------------------
    for k in prange(0, M, 1):

        for q in prange(k + 1, M, 1):

            RHO = 0;

            for i in prange(0, nc, 1):
                
                if (IF[i][k] < 2) : continue;

                for t in prange(0, M, 1) : v[t] = IF[i][t];

                sqrtOf = sqrt((v[k] - 1) * v[k] * (v[q] + 1) * (v[q] + 2));
                v[k] -= 2;
                v[q] += 2;
                j = FockToIndex(N, M, NCmat, v);
                RHO += C[i].conjugate() * C[j] * sqrtOf;

            # Use 2-index-'hermiticity'
            rho[k + k * M + q * M2 + q * M3] = RHO;
            rho[q + q * M + k * M2 + k * M3] = RHO.conjugate();
    # ---------------------------------------------------------------------



    # ---------------------------------------------
    # Rule 4: Creation on k k / Annihilation on k l
    # ---------------------------------------------------------------------
    for k in prange(0, M, 1):

        for l in prange(k + 1, M, 1):

            RHO = 0;

            for i in prange(0, nc, 1):

                if (IF[i][k] < 2) : continue;

                for t in prange(0, M, 1): v[t] = IF[i][t];

                sqrtOf = (v[k] - 1) * sqrt(v[k] * (v[l] + 1));
                v[k] -= 1;
                v[l] += 1;
                j = FockToIndex(N, M, NCmat, v);
                RHO += C[i].conjugate()* C[j] * sqrtOf;

            rho[k + k * M + k * M2 + l * M3] = RHO;
            rho[k + k * M + l * M2 + k * M3] = RHO;
            rho[l + k * M + k * M2 + k * M3] = RHO.conjugate();
            rho[k + l * M + k * M2 + k * M3] = RHO.conjugate();
    # ---------------------------------------------------------------------



    # ---------------------------------------------
    # Rule 5: Creation on k s / Annihilation on s s
    # ---------------------------------------------------------------------
    for k in prange(0, M, 1):

        for s in prange(k + 1, M, 1):

            RHO = 0;

            for i in prange(0, nc, 1):

                if (IF[i][k] < 1 or IF[i][s] < 1) : continue;

                for t in prange(0, M, 1): v[t] = IF[i][t];

                sqrtOf = v[s] * sqrt(v[k] * (v[s] + 1));
                v[k] -= 1;
                v[s] += 1;
                j = FockToIndex(N, M, NCmat, v);
                RHO += C[i].conjugate() * C[j] * sqrtOf;

            rho[k + s * M + s * M2 + s * M3] = RHO;
            rho[s + k * M + s * M2 + s * M3] = RHO;
            rho[s + s * M + s * M2 + k * M3] = RHO.conjugate();
            rho[s + s * M + k * M2 + s * M3] = RHO.conjugate();
    # ---------------------------------------------------------------------



    # -----------------------------------------------------------
    # Rule 6.0: Creation on k k / Annihilation on q l (k < q < l)
    # ---------------------------------------------------------------------
    for k in prange(0, M, 1):

        for q in prange(k + 1, M, 1):

            for l in prange(q + 1, M, 1):

                RHO = 0;

                for i in prange(0, nc, 1):
                    
                    if (IF[i][k] < 2) : continue;

                    for t in prange(0, M, 1): v[t] = IF[i][t];

                    sqrtOf = sqrt( v[k] * (v[k] - 1) * 
                             (v[q] + 1) * (v[l] + 1) );
                    v[k] -= 2;
                    v[l] += 1;
                    v[q] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    RHO += C[i].conjugate() * C[j] * sqrtOf;

                rho[k + k * M + q * M2 + l * M3] = RHO;
                rho[k + k * M + l * M2 + q * M3] = RHO;
                rho[l + q * M + k * M2 + k * M3] = RHO.conjugate();
                rho[q + l * M + k * M2 + k * M3] = RHO.conjugate();
    # ---------------------------------------------------------------------



    # -----------------------------------------------------------
    # Rule 6.1: Creation on k k / Annihilation on q l (q < k < l)
    # ---------------------------------------------------------------------
    for q in prange(0, M, 1):
        
        for k in prange(q + 1, M, 1):
            
            for l in prange(k + 1, M, 1):

                RHO = 0;

                for i in prange(0, nc, 1):
                    
                    if (IF[i][k] < 2) : continue;

                    for t in prange(0, M, 1): v[t] = IF[i][t];

                    sqrtOf = sqrt( v[k] * (v[k] - 1) * 
                             (v[q] + 1) * (v[l] + 1) );
                    v[k] -= 2;
                    v[l] += 1;
                    v[q] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    RHO += C[i].conjugate() * C[j] * sqrtOf;

                rho[k + k * M + q * M2 + l * M3] = RHO;
                rho[k + k * M + l * M2 + q * M3] = RHO;
                rho[l + q * M + k * M2 + k * M3] = RHO.conjugate();
                rho[q + l * M + k * M2 + k * M3] = RHO.conjugate();
    # ---------------------------------------------------------------------



    # -----------------------------------------------------------
    # Rule 6.2: Creation on k k / Annihilation on q l (q < l < k)
    # ---------------------------------------------------------------------
    for q in prange(0, M, 1):
        
        for l in prange(q + 1, M, 1):
            
            for k in prange(l + 1, M, 1):

                RHO = 0;

                for i in prange(0, nc, 1):
                    
                    if (IF[i][k] < 2) : continue;

                    for t in prange(0, M, 1): v[t] = IF[i][t];

                    sqrtOf = sqrt( v[k] * (v[k] - 1) * 
                             (v[q] + 1) * (v[l] + 1) );
                    v[k] -= 2;
                    v[l] += 1;
                    v[q] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    RHO += C[i].conjugate() * C[j] * sqrtOf;

                rho[k + k * M + q * M2 + l * M3] = RHO;
                rho[k + k * M + l * M2 + q * M3] = RHO;
                rho[l + q * M + k * M2 + k * M3] = RHO.conjugate();
                rho[q + l * M + k * M2 + k * M3] = RHO.conjugate();
    # ---------------------------------------------------------------------



    # -----------------------------------------------------------
    # Rule 7.0: Creation on k s / Annihilation on s l (s < k < l)
    # ---------------------------------------------------------------------
    for s in prange(0, M, 1):

        for k in prange(s + 1, M, 1):

            for l in prange(k + 1, M, 1):

                RHO = 0;

                for i in prange(0, nc, 1):
                    
                    if (IF[i][k] < 1 or IF[i][s] < 1) : continue;

                    for t in prange(0, M, 1) : v[t] = IF[i][t];
                    
                    sqrtOf = v[s] * sqrt(v[k] * (v[l] + 1));
                    v[k] -= 1;
                    v[l] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    RHO += C[i].conjugate() * C[j] * sqrtOf;

                rho[k + s * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + l * M2 + s * M3] = RHO;
                rho[k + s * M + l * M2 + s * M3] = RHO;

                rho[l + s * M + s * M2 + k * M3] = RHO.conjugate();
                rho[s + l * M + s * M2 + k * M3] = RHO.conjugate();
                rho[s + l * M + k * M2 + s * M3] = RHO.conjugate();
                rho[l + s * M + k * M2 + s * M3] = RHO.conjugate();
    # ---------------------------------------------------------------------



    # -----------------------------------------------------------
    # Rule 7.1: Creation on k s / Annihilation on s l (k < s < l)
    # ---------------------------------------------------------------------
    for k in prange(0, M, 1):

        for s in prange(k + 1, M, 1):

            for l in prange(s + 1, M, 1):

                RHO = 0;

                for i in prange(0, nc, 1):

                    if (IF[i][k] < 1 or IF[i][s] < 1) : continue;

                    for t in prange(0, M, 1) : v[t] = IF[i][t];

                    sqrtOf = v[s] * sqrt(v[k] * (v[l] + 1));
                    v[k] -= 1;
                    v[l] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    RHO += C[i].conjugate() * C[j] * sqrtOf;

                rho[k + s * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + l * M2 + s * M3] = RHO;
                rho[k + s * M + l * M2 + s * M3] = RHO;

                rho[l + s * M + s * M2 + k * M3] = RHO.conjugate();
                rho[s + l * M + s * M2 + k * M3] = RHO.conjugate();
                rho[s + l * M + k * M2 + s * M3] = RHO.conjugate();
                rho[l + s * M + k * M2 + s * M3] = RHO.conjugate();
    # ---------------------------------------------------------------------



    # -----------------------------------------------------------
    # Rule 7.2: Creation on k s / Annihilation on s l (k < l < s)
    # ---------------------------------------------------------------------
    for k in prange(0, M, 1):

        for l in prange(k + 1, M, 1):

            for s in prange(l + 1, M, 1):

                RHO = 0;

                for i in prange(0, nc, 1):

                    if (IF[i][k] < 1 or IF[i][s] < 1) : continue;

                    for t in prange(0, M, 1) : v[t] = IF[i][t];

                    sqrtOf = v[s] * sqrt(v[k] * (v[l] + 1));
                    v[k] -= 1;
                    v[l] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    RHO += C[i].conjugate() * C[j] * sqrtOf;

                rho[k + s * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + l * M2 + s * M3] = RHO;
                rho[k + s * M + l * M2 + s * M3] = RHO;

                rho[l + s * M + s * M2 + k * M3] = RHO.conjugate();
                rho[s + l * M + s * M2 + k * M3] = RHO.conjugate();
                rho[s + l * M + k * M2 + s * M3] = RHO.conjugate();
                rho[l + s * M + k * M2 + s * M3] = RHO.conjugate();
    # ---------------------------------------------------------------------



    # ---------------------------------------------
    # Rule 8: Creation on k s / Annihilation on q l
    # ---------------------------------------------------------------------
    for k in prange(0, M, 1):

        for s in prange(0, M, 1):

            if (s == k) : continue;

            for q in prange(0, M, 1):

                if (q == s or q == k) : continue;

                for l in prange(0, M, 1):

                    RHO = 0;

                    if (l == k or l == s or l == q) : continue;

                    for i in prange(0, nc, 1):
                        
                        if (IF[i][k] < 1 or IF[i][s] < 1) : continue;

                        for t in prange(0, M, 1) : v[t] = IF[i][t];

                        sqrtOf = sqrt(v[k] * v[s] * (v[q] + 1) * (v[l] + 1));
                        v[k] -= 1;
                        v[s] -= 1;
                        v[q] += 1;
                        v[l] += 1;
                        j = FockToIndex(N, M, NCmat, v);
                        RHO += C[i].conjugate() * C[j] * sqrtOf;

                    rho[k + s * M + q * M2 + l * M3] = RHO;

                # Finish l loop
            # Finish q loop
        # Finish s loop
    # Finish k loop
    # ---------------------------------------------------------------------

# END OF ROUTINE  ---------------------------------------------------------










###########################################################################
#                                                                         #
#                                                                         #
#                      ROUTINES TO COLLECT OBSERVABLES                    #
#                      ===============================                    #
#                                                                         #
#                                                                         #
###########################################################################










def GetOBrho(Npar, Morb, C):
    """
    CALLING : ( 2D array [Morb, Morb] ) = GetOBrho(Npar, Morb, C)
    -------
    return the one-body density matrix rho[k,l] = < a†ₖ aₗ >

    arguments :
    ---------
    Npar : # of particles
    Morb : # of orbitals (dimension of the matrix returned)
    C    : coefficients of configuration states basis
    """
    rho = np.empty([ Morb , Morb ], dtype=np.complex128);
    NCmat = GetNCmat(Npar, Morb);
    IF = GetFocks(Npar, Morb);
    OBrho(Npar, Morb, NCmat, IF, C, rho);
    return rho;





def GetTBrho(Npar, Morb, C):
    """
    CALLING : ( 1D array [ Morb⁴ ] ) = GetOBrho(Npar, Morb, C)
    -------
    return rho[k + s*Morb + n*Morb² + l*Morb³] = < a†ₖ a†ₛ aₙ aₗ >

    arguments :
    ---------
    Npar : # of particles
    Morb : # of orbitals (dimension of the matrix returned)
    C    : coefficients of configuration states basis
    """
    rho = np.empty(Morb * Morb * Morb * Morb, dtype=np.complex128);
    NCmat = GetNCmat(Npar, Morb);
    IF = GetFocks(Npar, Morb);
    TBrho(Npar, Morb, NCmat, IF, C, rho);
    return rho;





@jit( (int32, float64[:], complex128[:,:]) , nopython=True, nogil=True)

def EigSort(Nvals, RHOeigvals, RHOeigvecs):
    """
    Calling : (void) EigSort(Nvals, RHOeigvals, RHOeigvecs)
    -------
    Sort the order of eigenvalues to be decreasing and the order
    of columns of eigenvectors accordingly so that the  k column
    keep being the eigenvector of k-th eigenvalue.

    arguments :
    ---------
    Nvals      : dimension of rho = # of orbitals
    RHOeigvals : End up with eigenvalues in decreasing order
    RHOeigvecs : Change the order of columns accordingly to eigenvalues
    """
    auxR = 0.0;
    auxC = 0.0;
    for i in prange(1, Nvals, 1):
        j = i;
        while (RHOeigvals[j] > RHOeigvals[j-1] and j > 0):
            # Sort the vector
            auxR = RHOeigvals[j-1];
            RHOeigvals[j-1] = RHOeigvals[j];
            RHOeigvals[j] = auxR;
            # Sort the matrix
            for k in prange(0, Nvals, 1):
                auxC = RHOeigvecs[k][j-1];
                RHOeigvecs[k][j-1] = RHOeigvecs[k][j];
                RHOeigvecs[k][j] = auxC;
            j = j - 1;





def GetOccupation(Npar, Morb, C):
    """
    CALLING
    -------
    (1D array of size Morb) = GetOccupation(Npar, Morb, C)

    arguments
    ---------
    Npar : # of particles
    Morb : # of orbitals
    C    : coefficients of configurations
    """
    rho = GetOBrho(Npar, Morb, C);
    eigval, eigvec = la.eig(rho);
    EigSort(Morb, eigval.real, eigvec);
    return eigval.real / Npar;





def GetNatOrb(Npar, Morb, C, Orb):
    """
    CALLING
    -------
    ( 2D numpy array [Morb x Mpos] ) = GetNatOrb(Npar, Morb, C, Orb)

    arguments
    ---------
    Npar : # of particles
    Morb : # of orbitals
    C    : coefficients of many-body state in condiguration basis
    Orb  : Matrix with each row a 'working' orbital
    """
    rho = GetOBrho(Npar, Morb, C);
    eigval, eigvec = la.eig(rho);
    EigSort(Morb, eigval.real, eigvec);
    return np.matmul(eigvec.conj().T, Orb);





def GetGasDensity(NOocc, NO):
    """
    CALLING
    -------
    ( 1D np array [Mpos] ) = GetGasDensity(NOocc, NO)

    arguments
    ---------
    NOocc : natural occupations given by GetOccupation
    NO    : Natural orbitals given by GetNatOrb
    """
    return np.matmul(NOocc, abs(NO)**2);





def GetGasMomentumDensity(NOocc, NO, dx, bound='zero'):
    """
    CALLING
    -------
    ( 1D array freq, 1D array density ) = GetGasDensity(NOocc, NO)

    arguments
    ---------
    NOocc : natural occupations given by GetOccupation
    NO    : Natural orbitals given by GetNatOrb
    """
    # grid factor to extent the domain without changing the
    # position grid step, to improve resolution in momentum
    # space(reduce the momentum grid step).
    # Shall be an odd number in order to keep  the symmetry
    gf = 15;

    Morb = NO.shape[0];
    Mpos = NO.shape[1];

    if (bound == 'zero') :

        extNO = np.zeros([Morb,gf*Mpos],dtype=np.complex128);

        for i in range(Morb):
            l = int( (gf - 1) / 2 );
            k = int( (gf + 1) / 2 );
            extNO[i,l*Mpos:k*Mpos] = NO[i];

    else :

        extNO = np.zeros([Morb,gf*(Mpos-1)],dtype=np.complex128);

        for i in range(Morb):
            for k in range(gf):
                extNO[i,k*(Mpos-1):(k+1)*(Mpos-1)] = NO[i,:-1];

    NOfft = np.zeros(extNO.shape,dtype=np.complex128);

    for i in range(Morb): k, NOfft[i] = DnormFFT(dx, extNO[i]);

    denfft = GetGasDensity(NOocc, NOfft);

    return k , denfft;





def TimeOccupation(Npar, Morb, rhotime):
    """
    CALLING :
    -------
    ( 2D numpy array [Nsteps, Morb] ) = TimeOccupation(Morb, Nsteps, rhotime)

    arguments :
    ---------
    Npar    : # of particles
    Morb    : # of orbitals (# of columns in rhotime)
    rhotime : each line has row-major matrix representation (a vector)
              that is the  one-body  densiity matrix at each time-step
    """
    Nsteps = rhotime.shape[0];
    eigval = np.empty( [Nsteps, Morb] , dtype=np.complex128 );
    for i in range(Nsteps):
        eigval[i], eigvec = la.eig( rhotime[i].reshape(Morb, Morb) );
        EigSort(Morb, eigval[i].real, eigvec);
    return eigval / Npar;





@jit( (int32, int32, float64[:], complex128[:,:], complex128[:,:]) ,
      nopython=True, nogil=True)

def SpatialOBdensity(Morb, Mpos, NOoccu, NO, n):
    """ Inner loop of GetSpatialOBdensity """
    for i in prange(Mpos):
        for j in prange(Mpos):
            for k in prange(Morb):
                n[i,j] = n[i,j] + NOoccu[k] * NO[k,j].conjugate() * NO[k,i];





def GetSpatialOBdensity(NOoccu, NO):
    """
    CALLING
    -------
    ( 2d array [M,M] ) = SpatialOBdensity(occu, NO)

    arguments
    ---------
    NOoccu : occu[k] has # of particles occupying NO[k,:] orbital
    NO     : [Morb,M] matrix with each row being a natural orbital

    comments
    --------
    given two discretized positions Xi and Xj then the entries of
    the matrix returned (let me call n) represent:

    n[i,j] = <  Ψ†(Xj) Ψ(Xi)  >
    """
    Mpos = NO.shape[1];
    Morb = NO.shape[0];
    n = np.zeros([ Mpos , Mpos ], dtype=np.complex128);
    SpatialOBdensity(Morb, Mpos, NOoccu, NO, n);
    return n;





@jit( (int32, int32, float64[:], complex128[:,:], float64[:], float64[:,:]),
      nopython=True, nogil=True)

def OBcorre(Morb, Mpos, NOoccu, NO, GasDen, g):
    """ Inner loop of GetOBcorrelation """
    OrbSum = 0.0;
    for i in prange(Mpos):
        for j in prange(Mpos):
            OrbSum = 0.0;
            for k in prange(Morb):
                OrbSum = OrbSum + NOoccu[k] * NO[k,j].conjugate() * NO[k,i];
            g[i,j] = abs(OrbSum) / sqrt(GasDen[i] * GasDen[j]);





def GetOBcorrelation(NOocc, NO):
    """
    CALLING
    -------
    ( 2d array [M,M] ) = GetOBcorrelation(occu, NO)

    arguments
    ---------
    NOoccu : occu[k] has # of particles occupying NO[k,:] orbital
    NO     : [Morb,M] matrix with each row being a natural orbital

    comments
    --------
    given two discretized positions Xi and Xj then the entries of
    the matrix returned (let me call g) represent:

    g[i,j] = | <  Ψ†(Xj) Ψ(Xi)  > |  /  sqrt( Den(Xj) * Den(Xi) )
    """
    GasDensity = np.matmul(NOocc, abs(NO)**2);
    Morb = NO.shape[0];
    Mpos = NO.shape[1];
    g = np.empty( [Mpos , Mpos], dtype=np.float64 );
    OBcorre(Morb, Mpos, NOocc, NO, GasDensity, g);
    return g;





def GetOB_momentum_correlation(NOocc, NO, dx, bound='zero'):
    """
    CALLING
    -------
    ( 2d array [M,M] ) = GetOBcorrelation(occu, NO)

    arguments
    ---------
    NOoccu : occu[k] has # of particles occupying NO[k,:] orbital
    NO     : [Morb,M] matrix with each row being a natural orbital

    comments
    --------
    given two discretized positions Xi and Xj then the entries of
    the matrix returned (let me call g) represent:

    g[i,j] = | <  Ψ†(Xj) Ψ(Xi)  > |  /  sqrt( Den(Xj) * Den(Xi) )
    """

    Morb = NO.shape[0];
    Mpos = NO.shape[1];

    # grid factor to extent the domain without changing the
    # position grid step, to improve resolution in momentum
    # space(reduce the momentum grid step).
    # Shall be an odd number in order to keep  the symmetry
    gf = 15;

    Morb = NO.shape[0];
    Mpos = NO.shape[1];

    if (bound == 'zero') :

        extNO = np.zeros([Morb,gf*Mpos],dtype=np.complex128);

        for i in range(Morb):
            l = int( (gf - 1) / 2 );
            k = int( (gf + 1) / 2 );
            extNO[i,l*Mpos:k*Mpos] = NO[i];

    else :

        extNO = np.zeros([Morb,gf*(Mpos-1)],dtype=np.complex128);

        for i in range(Morb):
            for k in range(gf):
                extNO[i,k*(Mpos-1):(k+1)*(Mpos-1)] = NO[i,:-1];

    NOfft = np.zeros(extNO.shape,dtype=np.complex128);

    for i in range(Morb): k, NOfft[i] = L2normFFT(dx, extNO[i]);

    denfft = GetGasDensity(NOocc, NOfft);

    g = np.empty( [k.size , k.size], dtype=np.float64 );
    OBcorre(Morb, k.size, NOocc, NOfft, denfft, g);

    return k, g, denfft;





@jit( (int32, int32, complex128[:], complex128[:,:], complex128[:,:]),
      nopython=False, nogil=True)
def MutualProb(Morb,Mpos,rho2,S,mutprob):

    M  = Morb;
    M2 = Morb * Morb;
    M3 = Morb * M2;

    r2 = complex(0);
    o = complex(0);
    Sum = complex(0);

    for i in prange(Mpos):
        for j in prange(Mpos):

            Sum = 0;

            for k in prange(Morb):
                for l in prange(Morb):
                    for q in prange(Morb):
                        for s in prange(Morb):
                            r2 = rho2[k+l*M+q*M2+s*M3];
                            o = S[q,j]*S[s,i]*(S[k,j]*S[l,i]).conjugate();
                            Sum = Sum + r2 * o;

            mutprob[i,j] = Sum;





def GetTBcorrelation(Npar, Morb, C, S):
    """
    CALLING
    -------
    ( 2d array [Mpos,Mpos] ) = GetTBcorrelation(Npar,Morb,C,S)
    where Mpos means the number of discretized positions = S.shape[1]

    arguments
    ---------
    Npar : number of particles
    Morb : number of orbitals
    C    : array of coeficients
    S    : Working orbitals organized by rows

    comments
    --------
    The two-body correlation here is the probability to find two
    particles at (discretized)positions Xi and Xj divided by the
    probability to find one-particle independently Xi and another
    at Xj.
    """
    Mpos = S.shape[1];
    NOocc = GetOccupation(Npar, Morb, C);
    NO = GetNatOrb(Npar, Morb, C, S);
    den = GetGasDensity(NOocc, NO);

    rho2 = GetTBrho(Npar, Morb, C) / Npar / (Npar - 1);
    mutprob = np.zeros([Mpos,Mpos], dtype=np.complex128);
    MutualProb(Morb,Mpos,rho2,S,mutprob);

    g2 = np.zeros([Mpos,Mpos],dtype=np.complex128);

    for i in range(Mpos):
        for j in range(Mpos):
            g2[i,j] = mutprob[i,j];

    return g2;





def GetTB_momentum_corr(Npar, Morb, C, S, dx, bound = 'zero'):
    """
    CALLING
    -------
    freqs, 2d array [Mpos,Mpos] = GetTB_momentum_corr(Npar,Morb,C,S,dx)
    where Mpos means the number of discretized positions = S.shape[1]

    arguments
    ---------
    Npar : number of particles
    Morb : number of orbitals
    C    : array of coeficients
    S    : Working orbitals organized by rows
    dx   : grid step (sample spacing)
    (optional) bound : can be 'zero' or 'periodic'
    """

    Mpos = S.shape[1];
    NOocc = GetOccupation(Npar, Morb, C);
    NO = GetNatOrb(Npar, Morb, C, S);

    # grid factor to extent the domain without changing the
    # position grid step, to improve resolution in momentum
    # space(reduce the momentum grid step).
    # Shall be an odd number in order to keep  the symmetry

    gf = 7;

    if (bound == 'zero') :

        extS  = np.zeros([Morb,gf*Mpos],dtype=np.complex128);
        extNO = np.zeros([Morb,gf*Mpos],dtype=np.complex128);

        for i in range(Morb):
            l = int( (gf - 1) / 2 );
            k = int( (gf + 1) / 2 );
            extS[i,l*Mpos:k*Mpos] = S[i];
            extNO[i,l*Mpos:k*Mpos] = NO[i];

        Sfft = np.zeros(extS.shape,dtype=np.complex128);
        NOfft = np.zeros(extNO.shape,dtype=np.complex128);

        for i in range(Morb):
            k, Sfft[i] = L2normFFT(dx, extS[i]);
            k, NOfft[i] = L2normFFT(dx, extNO[i]);

    else :

        extS  = np.zeros([Morb,gf*(Mpos-1)],dtype=np.complex128);
        extNO = np.zeros([Morb,gf*(Mpos-1)],dtype=np.complex128);

        for i in range(Morb):
            for k in range(gf):
                init = k * (Mpos - 1);
                final = (k + 1) * (Mpos - 1);
                extS[i,init:final] = S[i,:Mpos-1];
                extNO[i,init:final] = NO[i,:Mpos-1];

        Sfft = np.zeros(extS.shape,dtype=np.complex128);
        NOfft = np.zeros(extNO.shape,dtype=np.complex128);

        for i in range(Morb):
            k, Sfft[i] = DnormFFT(dx, extS[i]);
            k, NOfft[i] = DnormFFT(dx, extNO[i]);

    denfft = GetGasDensity(NOocc, NOfft);

    rho2 = GetTBrho(Npar, Morb, C) / Npar / (Npar - 1);
    mutprob = np.zeros([k.size,k.size], dtype=np.complex128);
    MutualProb(Morb,k.size,rho2,Sfft,mutprob);

    g2 = np.zeros([k.size,k.size],dtype=np.complex128);

    for i in range(k.size):
        for j in range(k.size): g2[i,j] = mutprob[i,j];

    return k, g2, denfft;





def Cutoffmomentum(k,g2,denfft):

    likely = denfft.max();

    i = 0;
    while (denfft[i] / likely < 1E-3) : i = i + 1;

    j = denfft.size - 1;
    while (denfft[j] / likely < 1E-3) : j = j - 1;

    if ( abs(k[i]) >= abs(k[j]) ) :
        m = i;
        n = k.size - 1;
        while( k[n] >= abs(k[i]) ) : n = n - 1;
        n = n + 5; # Add little extra margin
        m = m - 4;

    else :
        n = j;
        m = 0;
        while( abs(k[m]) >= k[n]) : m = m + 1;
        m = m - 5;
        n = n + 4;

    return k[m:n], g2[m:n,m:n];





@jit( complex128(int32,complex128[:],complex128[:,:],complex128[:,:]),
      nopython=False, nogil=True)
def TB_Variance_part(Morb,rho2,Pmat,Qmat):

    M  = Morb;
    M2 = Morb * Morb;
    M3 = Morb * M2;

    r2 = complex(0);
    Sum = complex(0);

    for k in prange(Morb):
        for l in prange(Morb):
            for q in prange(Morb):
                for s in prange(Morb):
                    r2 = rho2[k+l*M+q*M2+s*M3];
                    Sum = Sum + r2 * Pmat[k,q] * Qmat[l,s];

    return Sum;





def GetMomentumVariance(Npar, Morb, C, S, dx):
    NOocc = GetOccupation(Npar, Morb, C);
    NO = GetNatOrb(Npar, Morb, C, S);

    rho2 = GetTBrho(Npar, Morb, C) / Npar / Npar;

    PMat = np.empty([Morb,Morb],dtype=np.complex128)

    for i in range(Morb):
        dSdx = derivative(dx,S[i]);
        PMat[i,i] = -1.0j * simps(S[i].conj() * dSdx, dx = dx);
        for j in range(i + 1,Morb):
            dSdx = derivative(dx,S[j]);
            PMat[i,j] = -1.0j * simps(S[i].conj() * dSdx, dx = dx);
            PMat[j,i] = PMat[i,j].conj();

    TBpart = TB_Variance_part(Morb,rho2,PMat,PMat);

    rho1 = GetOBrho(Npar,Morb,C) / Npar / Npar;

    P2mat = np.matmul(PMat,PMat);

    OBpart = 0;

    for i in range(Morb):
        for j in range(Morb): OBpart = OBpart + rho1[i,j]*P2mat[i,j];

#    for i in range(Morb):
#        dNOdx = derivative(dx,NO[i]);
#        OBpart = OBpart + simps(abs(dNOdx)**2,dx=dx)*NOocc[i];

#    OBpart = OBpart / Npar;

    avg = 0.0;
    for i in range(Morb):
        dNOdx = derivative(dx,NO[i]);
        avg = avg - 1.0j*simps(NO[i].conj()*dNOdx,dx=dx)*NOocc[i];

    return TBpart + OBpart - avg**2;





def GetMomentumVarianceFFT(Npar, Morb, C, S, dx):
    Mpos = S.shape[1];
    NOocc = GetOccupation(Npar, Morb, C);
    NO = GetNatOrb(Npar, Morb, C, S);

    rho2 = GetTBrho(Npar, Morb, C) / Npar / Npar;

    Sfft = np.zeros([Morb,Mpos-1],dtype=np.complex128);
    NOfft = np.zeros([Morb,Mpos-1],dtype=np.complex128);

    for i in range(Morb):
        k, Sfft[i] = DnormFFT(dx, S[i,:-1]);
        k, NOfft[i] = DnormFFT(dx, NO[i,:-1]);

    mutprob = np.zeros([k.size,k.size], dtype=np.complex128);
    MutualProb(Morb,k.size,rho2,Sfft,mutprob);

    TBpart = 0;

    for i in range(k.size):
        for j in range(k.size):
            TBpart = TBpart + k[i] * k[j] * mutprob[i,j];

    OBpart = 0.0;

    for i in range(Morb):
        dNOdx = derivative(dx,NO[i]);
        OBpart = OBpart + simps(abs(dNOdx)**2,dx=dx)*NOocc[i];

    OBpart = OBpart / Npar;

    avg = 0.0;
    for i in range(Morb):
        dNOdx = derivative(dx,NO[i]);
        avg = avg - 1.0j*simps(NO[i].conj()*dNOdx,dx=dx)*NOocc[i];

    return TBpart + OBpart - avg**2;





def GetMomentumCF_covariance(Npar, Morb, C, S, dx):
    NOocc = GetOccupation(Npar, Morb, C);
    NO = GetNatOrb(Npar, Morb, C, S);

    rho2 = GetTBrho(Npar, Morb, C) / Npar / Npar;

    pMat = np.empty([Morb,Morb],dtype=np.complex128);
    projMat = np.empty([Morb,Morb],dtype=np.complex128);

    for i in range(Morb):

        dSdx = derivative(dx,S[i]);
        pMat[i,i] = -1.0j * simps(S[i].conj()*dSdx,dx=dx);
        projMat[i,i] = abs( simps(S[i].conj()*NO[0],dx=dx) )**2;

        for j in range(i + 1,Morb):
            dSdx = derivative(dx,S[j]);
            pMat[i,j] = -1.0j * simps(S[i].conj() * dSdx, dx = dx);
            pMat[j,i] = momMat[i,j].conj();
            integral1 = simps(S[i].conj()*NO[0],dx=dx);
            integral2 = simps(NO[0].conj()*S[j],dx=dx);
            projMat[i,j] = integral1 * integral2;
            projMat[j,i] = (integral1 * integral2).conj();

    TBpart = TB_Variance_part(Morb,rho2,pMat,projMat);

    rho1 = GetOBrho(Npar,Morb,C) / Npar / Npar;

    P_PROJ_mat = np.matmul(projMat,pMat);

    OBpart = 0.0j;

    for i in range(Morb):
        for j in range(Morb):
            OBpart = OBpart + rho1[i,j] * P_PROJ_mat[i,j];

    avgP = 0.0;
    for i in range(Morb):
        dNOdx = derivative(dx,NO[i]);
        avgP = avgP - 1.0j*simps(NO[i].conj()*dNOdx,dx=dx)*NOocc[i];

    up = TBterm + OBterm - avgP * NOocc[0];
    
    sigP2 = GetMomentumVariance(Npar,Morb,C,S,dx);

    sigN02 = TB_Variance_part(Morb,rho2,projMat,projMat);
    sigN02 = sigN02 - NOocc[0]*(NOocc[0] - 1.0/Npar);

    down = sqrt(sigN02 * sigP2);

    return up / down;





def VonNeumannS(occ):
    """ (double) = VonNeumannS(occupation_vector of size # of Orbitals) """
    N = occ.sum();
    return - ( (occ / N) * np.log(occ / N) ).sum();

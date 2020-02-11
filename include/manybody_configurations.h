#ifndef _manybody_configurations_h
#define _manybody_configurations_h

#include <limits.h>
#include "array_memory.h"
#include "array_operations.h"

#ifdef _OPENMP
    #include <omp.h>
#endif





long fac(int n);

int NC(int N, int M);
/* total Number of configurations - dimension of spanned space */





Iarray setupNCmat(int N, int M);

Iarray setupFocks(int N, int M);
/* Hashing table for Fock states to avoid IndexToFock routine */





void IndexToFock(int k, int N, int M, Iarray v);
/* Convert an configuration index into the corresponding
 * number occupation vector                           */





int FockToIndex(int N, int M, Iarray NCmat, Iarray v);
/* Convert a occupation number vector(v) into its coeficient index */





Iarray OneOneMap(int N, int M, Iarray NCmat, Iarray IF);
Iarray allocTwoTwoMap(int nc, int M, Iarray IF);
Iarray TwoTwoMap(int N, int M, Iarray NCmat, Iarray IF, Iarray strideC);
Iarray allocOneTwoMap(int nc, int M, Iarray IF);
Iarray OneTwoMap(int N, int M, Iarray NCmat, Iarray IF, Iarray strideC);
/* Auxiliar mappings. See the source file to read what are their utility */





void OBrho(int N, int M, Iarray Map, Iarray IF, Carray C, Cmatrix rho);
/* Construct the one-body density matrix */





void TBrho(int N, int M, Iarray Map, Iarray MapOT, Iarray MapTT,
     Iarray strideOT, Iarray strideTT, Iarray IF, Carray C, Carray rho);
/* Construct the two-body density matrix. The storage in memory is as
 * linearization as follows:
 * rho[k, l, s, q] = rho[k + l * M + s * M^2 + q * M^3]           **/





void applyHconf (int N, int M, Iarray Map, Iarray MapOT, Iarray MapTT,
     Iarray strideOT, Iarray strideTT, Iarray IF, Carray C, Cmatrix Ho,
     Carray Hint, Carray out);
/* Act with the Hamiltonian operator on a many-body state expressed
 * through its coefficients in the configurational basis.        */





#endif

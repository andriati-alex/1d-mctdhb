#ifndef _configurationalSpace_h
#define _configurationalSpace_h

#include "memoryHandling.h"
#include "array_operations.h"



long fac(int n);
// NC return the configurational space dimension
int NC(int Nparticles, int Norbitals);



// Define auxiliar array to avoid calls of NC
Iarray setupNCmat(int N, int M);
// Setup a Hashing table for Fock states to avoid IndexToFock routine 
Iarray setupFocks(int N, int M);



// Convert an configuration index into the number occupation vector
void IndexToFock(int k, int N, int M, Iarray v);
// Convert a occupation number vector(v) into its coefficient index
int FockToIndex(int N, int M, Iarray NCmat, Iarray v);



// Auxiliar mappings. See the source file to read about their utility
Iarray OneOneMap(int N, int M, Iarray NCmat, Iarray IF);
Iarray allocTwoTwoMap(int nc, int M, Iarray IF);
Iarray TwoTwoMap(int N, int M, Iarray NCmat, Iarray IF, Iarray strideC);
Iarray allocOneTwoMap(int nc, int M, Iarray IF);
Iarray OneTwoMap(int N, int M, Iarray NCmat, Iarray IF, Iarray strideC);



// Density matrices
// memory access for 2-body density matrix :
// rho[k, l, s, q] = rho[k + l * M + s * M^2 + q * M^3]
void OBrho(int N, int M, Iarray Map, Iarray IF, Carray C, Cmatrix rho);
void TBrho(int N, int M, Iarray Map, Iarray MapOT, Iarray MapTT,
     Iarray strideOT, Iarray strideTT, Iarray IF, Carray C, Carray rho);



// Action of the many-body Hamiltonian in a state represented by the
// coefficients C in the configurational basis
void applyHconf (int N, int M, Iarray Map, Iarray MapOT, Iarray MapTT,
     Iarray strideOT, Iarray strideTT, Iarray IF, Carray C, Cmatrix Ho,
     Carray Hint, Carray out);



#endif

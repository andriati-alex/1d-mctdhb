#ifndef _coef_routines_h
#define _coef_routines_h





#include "array_memory.h"
#include "array_operations.h"





#ifdef _OPENMP
    #include <omp.h>
#endif





int fac(int n);





int NC(int N, int M);
/* ************************************
 * Number of possible configurations of
 *
 * N identical particles
 * in M distinct orbital
 *
 * (N + M - 1)! / (N! * (M - 1)!)
 *
 * ************************************/





int ** MountNCmat(int N, int M);
/* ******************************************************
 *
 * Construct the matrix with  all possible configurations
 * up to N particles and M orbitals. A row index  is  the
 * number of particles and the column index is the number
 * of orbitals. The Matrix has of (N+1) x (M+1) dimension
 * and its values are NCmat[i, j] = NC( i , j )
 *
 * ******************************************************/





int ** MountFocks(int N, int M, int ** NCmat);
/* *************************************************************
 *
 * Give a matrix whose the row i contains the occupation numbers
 * of a possible configuration
 *
 * *************************************************************/





void IndexToFock(int k, int N, int M, int ** NCmat, int * v);
/* ************************************************************
 * 
 * Convert an coeficient index into a Fock's occupation vector
 *
 * Given N particles and M orbitals the total number of
 * coeficients is given by NC( N , M ), whereas to num-
 * ber out all coeficients we need 0 <= k < NC( N , M).
 * Starting  from the last orbital check if k is bigger
 * than the number of all possible combinations  of all
 * particles within all orbitals but the last one. That
 * is if k > NC( N , M - 1 ). In positive case we put a
 * particle in the orbital M. Thereafter we subtract 
 * from k the number NC( N , M - 1 ), a kind of a  cost
 * to put a particle in M,  and proceed with N - 1 par-
 * ticles. As soon as k is not bigger than the NC number
 * in a given iteration we try to put in the orbital to
 * the left, that is, reduce M -> M - 1.
 *
 * v end up with occupation vector related to k
 *
 * ***********************************************************/





int FockToIndex(int N, int M, int ** NCmat, int * v);
/* ****************************************************
 * 
 * Convert a Fock's occupation number vector(v) into its coeficient index
 *
 * Do the reverse job of IndexToFock. The algorithm
 * is design to take out of the vector all the par-
 * ticles, from the last to the first orbital adding
 * up the 'index cost' to put it there, described in
 * IndexToFock.
 *
 * Return the integer corresponding to the index
 *
 * ****************************************************/





int * JumpMapping(int N, int M, int ** NCmat, int ** IF);
/* ******************************************************************
 *
 * Make a mapping between the coeficient i to the coeficient that
 * has one particle less in orbital k and one more in l, given th
 * vector corresponding to Ci.
 *
 * To accomplish the task for every i, k and l, the pointer Map
 * arrange its elements in the given way:
 *
 * Map[i + nc * k + nc * M * l] = Index of C having (k ---> l)
 *
 * Where nc is the short for NC( N , M ).
 *
 * Map must have size = nc * M ^2.
 *
 * ******************************************************************/





int ** AllocTwiceMap(int N, int M, int ** IF);
int ** TwiceJumpMapping(int N, int M, int ** NCmat, int ** IF);






void OBrho(int N, int M, int * Map, int ** NCmat, int ** IF,
     Carray C, Cmatrix rho);
/* Construct the one-body density matrix given N particles and M orbitals
 * and the coefficients from the state in the occupation number basis **/





void TBrho(int N, int M, int * Map1, int ** Map2, int ** NCmat, int ** IF,
     Carray C, Carray rho);
/* Construct the two-body density matrix given N particles and M orbitals
 * and the coefficients from the state in the  occupation  number  basis.
 * The storage in memory follow a linearization as follows:
 * rho[k, l, s, q] = rho[k + l * M + s * M^2 + q * M^3]               **/





void applyHconf (int N, int M, int * Map1, int ** Map2, int ** NCmat,
     int ** IF, Carray C, Cmatrix Ho, Carray Hint, Carray out);
/* Give the state coefficients of a state (out) after apply the many-body
 * Hamiltonian on a state whose  coeficients  in  the  occupation  number
 * basis are C[i]. Ho contains the matrix elements of  one-body  operator
 * and Hint contains the matrix elements of two-body operator.        **/





#endif

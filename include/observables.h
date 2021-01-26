#ifndef _observables_h
#define _observables_h

#include "calculus.h"
#include "memoryHandling.h"



/* ========================================================================
 *                                                                *
 *        Setup matrix elements of one-body Hamiltonian Ho        *
 *        ------------------------------------------------        *
 *                                                                */

void SetupHo (int,int,Cmatrix,double,double,doublec,Rarray,Cmatrix);





/* ========================================================================
 *                                                                  *
 *        Setup matrix elements of two-body Hamiltonian Hint        *
 *        --------------------------------------------------        *
 *                                                                  */

void SetupHint (int,int,Cmatrix,double,double,Carray);





/* ========================================================================
 *                                                                  *
 *        Energy for a given set of orbitals and coeficients        *
 *        --------------------------------------------------        *
 *                                                                  */

doublec Energy (int,Cmatrix,Carray,Cmatrix,Carray);





doublec KinectE (int,int,Cmatrix,double,double,Cmatrix);





doublec PotentialE (int,int,Cmatrix,double,Rarray,Cmatrix);





doublec TwoBodyE (int,int,Cmatrix,double,double g,Carray);





doublec Virial(EqDataPkg,Cmatrix,Cmatrix,Carray);





double MeanQuadraticR(EqDataPkg,Cmatrix,Cmatrix);



#endif

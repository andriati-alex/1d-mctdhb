#ifndef _dataStructures_h
#define _dataStructures_h

/** DATATYPES OF THIS MCTDHB PACKAGE
    ================================

    Here are all relevant datatypes and structures for all others files.
    Among the external headers used the mkl is probably the one that is
    intalled separately, all the others are usual from C language.

    The structures are funcamental for clean organization of all relevant
    parameters in the MCTDHB problem and contains all that  is  needed to
    setup the numerical problem properly                              **/





// EXTERNAL HEADERS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <complex.h>
#include <mkl.h>
#include <mkl_dfti.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

// CONSTANTS
#define PI 3.141592653589793

// SHORTCUTS
typedef double complex doublec;
typedef int * Iarray;
typedef int ** Imatrix;
typedef double * Rarray;
typedef double ** Rmatrix;
typedef double complex * Carray;
typedef double complex ** Cmatrix;
typedef MKL_Complex16 * CMKLarray;



// STRUCTURES TO COMPRESS RELEVANT DATA TO SOLVE THE EQUATIONS
struct _EquationDataPkg
{

    char
        Vname[80]; // One-Body potential name

    int 
        nc,       // Total # of configurations
        Mpos,     // # of grid points (# grid steps + 1)
        Morb,     // # of orbitals
        Npar;     // # of particles

    Iarray
        IF,    // IF[i*Morb] point to the occupation number vetor of C[i]
        NCmat, // NCmat[i + j*(Npar+1)] = NC(i,j)
        Map,   // Mappings for jumps of particles among orbitals
        MapOT,
        MapTT,
        strideOT,
        strideTT;

    double
        dx, // grid step size
        xi, // left boundary of domain (first grid point)
        xf, // right boundaty of domain (last grid point)
        a2, // factor multiplying d2 / dx2
        g;  // know as g, contact interaction strength

    Rarray
        V;  // One-body potential(trap) computed at grid points

    double
        p[3];   // Extra parameters to one-body potential function

    double complex
        a1;     // factor multiplying d / dx (pure imaginary)

};



struct _ManyBodyDataPkg
{

    int 
        nc,     // Total # of configurations
        Mpos,   // # of grid points (# grid steps + 1)
        Morb,   // # of orbitals
        Npar;   // # of particles

    Carray
        C,      // Vector of coefficients in the configurational space
        Hint,   // Matrix elements of two-body part of hamiltonian
        rho2;   // two-body density matrix

    Cmatrix
        Ho,   // Matrix elements of one-body part of hamiltonian
        Omat, // Matrix of orbitals(one per row) at grid points
        rho1; // one-body density matrix

};



struct _CCSmat{
    int  m;     // max number of non-zero elements in a same row
	int * col;  // Column index of elemetns.
	Carray vec; // column oriented vector.
};



// SHORTCUT TO STRUCTURE POINTERS
typedef struct _ManyBodyDataPkg * ManyBodyPkg;
typedef struct _EquationDataPkg * EqDataPkg;
typedef struct _CCSmat * CCSmat;

#endif

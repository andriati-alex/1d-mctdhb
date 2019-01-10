#ifndef _data_structure_h
#define _data_structure_h

#include <string.h>
#include "linear_potential.h"
#include "manybody_configurations.h"



/* ======================================================================== */
/*                                                                          */
/*         DATA-TYPE DEFINITION - All information Needed for MCTDHB         */
/*                                                                          */
/* ======================================================================== */



struct _EquationDataPkg
{

    char
        Vname[80]; // One-Body potential

    int 
        nc,       // Total # of configurations(Fock states)
        Mpos,     // # of discretized positions (# divisions + 1)
        Morb,     // # of orbitals
        Npar,     // # of particles
        ** IF,    // IF[i] point to the occupation number vetor of C[i]
        ** NCmat; // NCmat[n][m] # with n particles / m orbitals

    double
        dx,       // space step
        xi,       // initial position discretized value
        xf,       // final position discretized value
        a2,       // factor multiplying d2 / dx2
        inter,    // know as g, contact interaction strength
        * V;      // Array with the values of one-particle potential

    double
        p[3];     // Parameters to generate one-particle potential values

    double complex
        a1;       // factor multiplying d / dx (pure imaginary)

};

typedef struct _EquationDataPkg * EqDataPkg;





struct _ManyBodyDataPkg
{

    int
        nc,   // number of possible configurations
        Mpos, // # of discretized positions (# divisions + 1)
        Morb, // # of orbitals
        Npar; // # of particles

    Carray
        C,    // Vector of coefficients for each configuration
        Hint, // Matrix elements of two-body part of hamiltonian
        rho2; // two-body density matrix

    Cmatrix
        Ho,   // Matrix elements of one-body part of hamiltonian
        Omat, // Matrix of orbitals(one per row) in discretized positions
        rho1; // one-body density matrix

};

typedef struct _ManyBodyDataPkg * ManyBodyPkg;





/* ======================================================================== *
 *                                                                          *
 *                           FUNCTION PROTOTYPES                            *
 *                                                                          *
 * ======================================================================== */





EqDataPkg PackEqData(int,int,int,double,double,double,double,doublec,
          char [],double []);

void dpkgEqData(EqDataPkg, double *, doublec *, Rarray, double *);

ManyBodyPkg AllocManyBodyPkg(int,int,int);

void ReleaseManyBodyDataPkg (ManyBodyPkg);

void ReleaseEqDataPkg (EqDataPkg);



#endif

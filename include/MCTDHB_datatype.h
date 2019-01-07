#ifndef _MCTDHB_datatype_h
#define _MCTDHB_datatype_h

#include <complex.h>
#include "MCTDHB_configurations.h"



/* ======================================================================== */
/*                                                                          */
/*         DATA-TYPE DEFINITION - All information Needed for MCTDHB         */
/*                                                                          */
/* ======================================================================== */



struct _MCTDHBsetup
{

    int 
        nc,       // Total # of configurations(Fock states)
        Mpos,     // # of discretized positions (# divisions + 1)
        Morb,     // # of orbitals
        Npar,     // # of particles
        ** IF,    // IF[i] point to the occupation number vetor of C[i]
        ** NCmat; // NCmat[n][m] # with n particles / m orbitals

    double
        dx,     // space step
        xi,     // initial position discretized value
        xf,     // final position discretized value
        a2,     // factor multiplying d2 / dx2
        inter,  // know as g, contact interaction strength
        * V;    // Array with the values of one-particle potential

    double complex
        a1;     // factor multiplying d / dx (pure imaginary)

};

typedef struct _MCTDHBsetup * MCTDHBsetup;





struct _MCTDHBmaster
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

typedef struct _MCTDHBmaster * MCTDHBmaster;





/* ======================================================================== *
 *                                                                          *
 *                           FUNCTION PROTOTYPES                            *
 *                                                                          *
 * ======================================================================== */





MCTDHBsetup AllocMCTDHBdata (int Npar,int Morb,int Mpos,double xi,double xf,
            double a2,double inter,double * V,double complex a1);

MCTDHBmaster AllocMCTDHBmaster (int Npar,int Morb,int Mpos);

void EraseMCTDHBmaster (MCTDHBmaster);

void EraseMCTDHBdata (MCTDHBsetup);



#endif

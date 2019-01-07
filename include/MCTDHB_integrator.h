#ifndef _MCTDHB_integrator_h
#define _MCTDHB_integrator_h

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <string.h>

#include "MCTDHB_datatype.h"
#include "MCTDHB_observables.h"
#include "MCTDHB_configurations.h"
#include "matrix_operations.h"
#include "tridiagonal_solver.h"



/* ======================================================================== */
/*                                                                          */
/*                           FUNCTION PROTOTYPES                            */
/*                                                                          */
/* ======================================================================== */





/* ==========================================================================
 *                                                                   *
 *          Function to apply nonlinear part of obitals PDE          *
 *          -----------------------------------------------          *
 *                                                                   */





double complex nonlinear (int M, int k, int n, double g, Cmatrix Orb,
               Cmatrix Rinv, Carray R2, Cmatrix Ho, Carray Hint );
/* For a orbital 'k' computed at discretized position 'n' calculate
   the right-hand-side part of MCTDHB orbital's equation of  motion
   that is nonlinear, part because of projections that made the eq.
   an integral-differential equation, and other part due to contact
   interactions. Assume that Rinv, R2 are  defined  by  the  set of
   configuration-state coefficients as the inverse of  one-body and
   two-body density matrices respectively. Ho and Hint are  assumed
   to be defined accoding to 'Orb' variable as well.            **/





/* ==========================================================================
 *                                                                   *
 *                    Time Evolution of equations                    *
 *                    ---------------------------                    *
 *                                                                   */





void MCNLTRAP_dOdt (MCTDHBsetup MC, Cmatrix Orb, Cmatrix dOdt,
     Cmatrix Ho, Carray Hint, Cmatrix rho1, Carray rho2);





void MCNL_dOdt (MCTDHBsetup MC, Cmatrix Orb, Cmatrix dOdt,
     Cmatrix Ho, Carray Hint, Cmatrix rho1, Carray rho2);





void MC_dCdt
(   // Compute time derivative of coefficient
    MCTDHBsetup MC,
    Carray C,
    Cmatrix Ho,
    Carray Hint,
    Carray dCdt
);





int lanczos(MCTDHBsetup MCdata, Cmatrix Ho, Carray Hint,
    int lm, Carray diag, Carray offdiag, Cmatrix lvec);





double LanczosGround (int Niter, MCTDHBsetup MC, Cmatrix Orb, Carray C);





void LanczosIntegrator
(   // Evolve nonlinear part of orbitals coupled with coeficients
    MCTDHBsetup MC,
    Cmatrix orb, // End up modified by the evolution
    Carray C,    // End up modified by the evolution
    double complex dt
);





void MC_NLTRAPC_IRK4 (MCTDHBsetup MC, MCTDHBmaster S, double complex dt);




/*
void MC_NL_IRK4
(   // Evolve nonlinear part with imaginary time
    MCTDHBsetup MC,
    Cmatrix Orb,
    Carray C,
    double complex dt
);
*/





void MC_NLC_IRK4 (MCTDHBsetup MC, MCTDHBmaster S, double complex dt);





void MCLP_CNSM
(   // Evolve a time-step the linear part of PDE(orbitals)
    int Mpos,
    int Morb,
    CCSmat cnmat, // Matrix from CN approach(discretization)
    Carray upper, // Upper diagonal
    Carray lower, // Lower diagonal
    Carray mid,   // Main diagonal of tridiagonal system
    Cmatrix Orb
);





void MCLP_CNLU (int Mpos, int Morb, CCSmat cnmat, Carray upper,
     Carray lower, Carray mid, Cmatrix Orb);





void MCLP_FFT (int Mpos, int Morb, DFTI_DESCRIPTOR_HANDLE * desc,
     Carray exp_der, Cmatrix Orb);





/* ==========================================================================
 *                                                                   *
 *                     Main routine to be called                     *
 *                    ---------------------------                    *
 *                                                                   */





int MC_IMAG_RK4_FFTRK4 (MCTDHBsetup, MCTDHBmaster, Carray E,
     Carray virial, double dT, int Nsteps);





int MC_IMAG_RK4_CNSMRK4 (MCTDHBsetup, MCTDHBmaster , Carray E,
     Carray virial, double dT, int Nsteps, int cyclic);





int MC_IMAG_RK4_CNLURK4 (MCTDHBsetup, MCTDHBmaster, Carray E,
     Carray virial, double dT, int Nsteps, int cyclic);





#endif

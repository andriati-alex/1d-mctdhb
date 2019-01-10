#ifndef _integrator_routine_h
#define _integrator_routine_h

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <string.h>

#include "data_structure.h"
#include "observables.h"
#include "manybody_configurations.h"
#include "inout.h"
#include "matrix_operations.h"
#include "tridiagonal_solver.h"



/* ======================================================================== */
/*                                                                          */
/*                           FUNCTION PROTOTYPES                            */
/*                                                                          */
/* ======================================================================== */



void ResizeDomain(EqDataPkg mc, ManyBodyPkg S);
/** In case of trapped system resize domain if needed **/



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





void NLTRAP_dOdt(EqDataPkg, Cmatrix , Cmatrix, Cmatrix, Carray, Cmatrix,
     Carray);





void NL_dOdt(EqDataPkg, Cmatrix, Cmatrix, Cmatrix, Carray, Cmatrix, Carray);





void dCdt(EqDataPkg, Carray, Cmatrix, Carray, Carray);





int lanczos(EqDataPkg MCdata, Cmatrix Ho, Carray Hint,
    int lm, Carray diag, Carray offdiag, Cmatrix lvec);





double LanczosGround (int Niter, EqDataPkg MC, Cmatrix Orb, Carray C);





void LanczosIntegrator(EqDataPkg, Cmatrix, Carray, double complex);





void NL_TRAP_C_RK4 (EqDataPkg, ManyBodyPkg, double complex);





void NL_C_RK4 (EqDataPkg, ManyBodyPkg, double complex);





void LP_CNSM(int, int, CCSmat, Carray, Carray, Carray, Cmatrix);





void LP_CNLU(int, int, CCSmat, Carray, Carray, Carray, Cmatrix);





void LP_FFT (int, int, DFTI_DESCRIPTOR_HANDLE *, Carray, Cmatrix);





/* ==========================================================================
 *                                                                   *
 *                     Main routine to be called                     *
 *                    ---------------------------                    *
 *                                                                   */





int IMAG_RK4_FFTRK4(EqDataPkg, ManyBodyPkg, Carray, Carray, double, int);





int IMAG_RK4_CNSMRK4(EqDataPkg,ManyBodyPkg,Carray,Carray,double,int,int);





int IMAG_RK4_CNLURK4(EqDataPkg,ManyBodyPkg,Carray,Carray,double,int,int);



#endif

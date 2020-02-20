#ifndef _integrators_h
#define _integrators_h

#include "linear_potential.h"
#include "dataStructures.h"
#include "observables.h"
#include "configurationalSpace.h"
#include "inout.h"
#include "matrix_operations.h"
#include "tridiagonal_solver.h"
#include "interpolation.h"



/*************************************************************************
 ********************                                 ********************
 ********************        AUXILIAR ROUTINES        ********************
 ********************                                 ********************
 *************************************************************************/

int NonVanishingId(int,Carray,double,double);

void ResizeDomain(EqDataPkg,ManyBodyPkg);

double overlapFactor(int,int,double,Cmatrix);

doublec nonlinear(int,int,int,double,Cmatrix,Cmatrix,Carray,Cmatrix,Carray);

doublec nonlinearOrtho(int,int,int,double,Cmatrix,Cmatrix,
        Carray,Cmatrix,Carray,Cmatrix);



/**************************************************************************
 **********                                                      **********
 **********   TIME-DERIVATIVES OF NONLINEAR PART OF SPLIT-STEP   **********
 **********                                                      **********
 **************************************************************************/

void imagNLTRAP_dOdt(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Cmatrix,Carray);

void imagNL_dOdt(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Cmatrix,Carray);

void realNL_dOdt(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Cmatrix,Carray);

void realNLTRAP_dOdt(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Cmatrix,Carray);



/**************************************************************************
 **************                                            ****************
 **************    COEFFICIENT INTEGRATION/GROUND-STATE    ****************
 **************                                            ****************
 **************************************************************************/

void dCdt(EqDataPkg,Carray,Cmatrix,Carray,Carray);

int lanczos(EqDataPkg,Cmatrix,Carray,int,Carray,Carray,Cmatrix);

double LanczosGround(int,EqDataPkg,Cmatrix,Carray);

void LanczosIntegrator(int,EqDataPkg,Cmatrix,Carray,doublec,Carray);

void coef_RK4(EqDataPkg,ManyBodyPkg,doublec);



/*************************************************************************
 **************                                             **************
 **************    RUNGE-KUTTA INTEGRATOS NONLINEAR PART    **************
 **************                                             **************
 *************************************************************************/

void imagNLTRAP_RK2(EqDataPkg,ManyBodyPkg,doublec);

void imagNL_RK2(EqDataPkg,ManyBodyPkg,doublec);

void realNLTRAP_RK2(EqDataPkg,ManyBodyPkg,double);

void realNL_RK2(EqDataPkg,ManyBodyPkg,double);

void realNL_RK4(EqDataPkg,ManyBodyPkg,double);

void realNLTRAP_RK4(EqDataPkg,ManyBodyPkg,double);



/************************************************************************
 ****************                                        ****************
 ****************     GENERAL INTEGRATOS LINEAR PART     ****************
 ****************                                        ****************
 ************************************************************************/

void LP_CNSM(int,int,CCSmat,Carray,Carray,Carray,Cmatrix);

void LP_CNLU(int,int,CCSmat,Carray,Carray,Carray,Cmatrix);

void LP_FFT (int,int,DFTI_DESCRIPTOR_HANDLE *,Carray,Cmatrix);










/***********************************************************************
 ***********************************************************************
 ******************                                    *****************
 ******************       SPLIT-STEP INTEGRATORS       *****************
 ******************                                    *****************
 ***********************************************************************
 ***********************************************************************/

int imagFFT(EqDataPkg,ManyBodyPkg,double,int,int);

int imagCNSM(EqDataPkg,ManyBodyPkg,double,int,int,int);

int imagCNLU(EqDataPkg,ManyBodyPkg,double,int,int,int);

void realCNSM(EqDataPkg,ManyBodyPkg,double,int,int,char [],int);

void realFFT(EqDataPkg,ManyBodyPkg,double,int,char [],int);



#endif

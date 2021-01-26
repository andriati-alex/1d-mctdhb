#ifndef _coeffIntegration_h
#define _coeffIntegration_h

#include "dataStructures.h"
#include "observables.h"
#include "configurationalSpace.h"
#include "inout.h"



void dCdt(EqDataPkg,Carray,Cmatrix,Carray,Carray);

int lanczos(EqDataPkg,Cmatrix,Carray,int,Carray,Carray,Cmatrix);

double LanczosGround(int,EqDataPkg,Cmatrix,Carray);

void LanczosIntegrator(int,EqDataPkg,Cmatrix,Carray,doublec,Carray);

void coef_RK4(EqDataPkg,ManyBodyPkg,doublec);



#endif

#ifndef _linearPartIntegration_h
#define _linearPartIntegration_h

#include "dataStructures.h"
#include "matrix_operations.h"
#include "tridiagonal_solver.h"

void linearCN(EqDataPkg,Carray,Carray,Carray,Cmatrix,int,doublec);

void linearFFT(int,int,DFTI_DESCRIPTOR_HANDLE *,Carray,Cmatrix);

#endif

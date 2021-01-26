#ifndef _orbTimeDerivativeDVR_h
#define _orbTimeDerivativeDVR_h

#include "dataStructures.h"
#include "matrix_operations.h"
#include "calculus.h"
#include "inout.h"

void derSINEDVR(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Rarray,int);
void derEXPDVR(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Carray,int);
void imagderEXPDVR(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Carray);

#endif

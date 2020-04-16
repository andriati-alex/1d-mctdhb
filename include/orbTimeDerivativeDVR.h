#ifndef _orbTimeDerivativeDVR_h
#define _orbTimeDerivativeDVR_h

#include "dataStructures.h"
#include "matrix_operations.h"
#include "calculus.h"
#include "inout.h"

void derSINEDVR(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Rarray);
void derEXPDVR(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Carray);
void imagderEXPDVR(EqDataPkg,Cmatrix,Cmatrix,Cmatrix,Carray,Carray);

#endif

#ifndef _imagtimeIntegrator_h
#define _imagtimeIntegrator_h

#include "linear_potential.h"
#include "dataStructures.h"
#include "observables.h"
#include "configurationalSpace.h"
#include "inout.h"
#include "auxIntegration.h"
#include "coeffIntegration.h"
#include "linearPartIntegration.h"
#include "orbTimeDerivativeDVR.h"

int imagEXPDVR(EqDataPkg,ManyBodyPkg,double,int,int);
int imagSSFFT(EqDataPkg,ManyBodyPkg,double,int,int);
int imagSSFD(EqDataPkg,ManyBodyPkg,double,int,int);

#endif

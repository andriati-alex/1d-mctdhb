#ifndef _realtimeIntegrator_h
#define _realtimeIntegrator_h

#include "linear_potential.h"
#include "dataStructures.h"
#include "observables.h"
#include "configurationalSpace.h"
#include "inout.h"
#include "auxIntegration.h"
#include "coeffIntegration.h"
#include "linearPartIntegration.h"
#include "orbTimeDerivativeDVR.h"

void realSSFD(EqDataPkg,ManyBodyPkg,double,int,char [],int);
void realSSFFT(EqDataPkg,ManyBodyPkg,double,int,char [],int);
void realSINEDVR(EqDataPkg,ManyBodyPkg,double,int,char [],int);
void realEXPDVR(EqDataPkg,ManyBodyPkg,double,int,char [],int);

#endif

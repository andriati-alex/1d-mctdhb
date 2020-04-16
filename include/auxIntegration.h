#ifndef _auxIntegration_h
#define _auxIntegration_h

#include "dataStructures.h"
#include "observables.h"
#include "configurationalSpace.h"
#include "inout.h"
#include "interpolation.h"
#include "linear_potential.h"



/*************************************************************************
 ********************                                 ********************
 ********************        AUXILIAR ROUTINES        ********************
 ********************                                 ********************
 *************************************************************************/

void recorb_inline(FILE *,int,int,int,Cmatrix);

int NonVanishingId(int,Carray,double,double);

void ResizeDomain(EqDataPkg,ManyBodyPkg);

void extentDomain(EqDataPkg,ManyBodyPkg);

double overlapFactor(int,int,double,Cmatrix);

double avgOrbNorm(int,int,double,Cmatrix);

double borderNorm(int,int,Rarray,double);

double eigQuality(EqDataPkg,Carray,Cmatrix,Carray,double);

#endif

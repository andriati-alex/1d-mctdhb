#ifndef _structureSetup_h
#define _structureSetup_h

#include "configurationalSpace.h"
#include "linear_potential.h"

EqDataPkg PackEqData(int,int,int,double,double,double,
                     double,doublec,char [],double []);

ManyBodyPkg AllocManyBodyPkg(int,int,int);

#endif

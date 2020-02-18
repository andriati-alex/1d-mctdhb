#ifndef _memoryHandling_h
#define _memoryHandling_h

#include "dataStructures.h"



// ROUTINES TO ALLOCATE MEMORY

Iarray iarrDef(int);

Rarray rarrDef(int);

Carray carrDef(int);

CMKLarray cmklDef(int);

Rmatrix rmatDef(int,int);

Cmatrix cmatDef(int,int);

CCSmat ccsmatDef(int,int);

// ROUTINES TO FREE ALLOCATED MEMORY

void rarrFree(Rarray);

void carrFree(Carray);

void iarrFree(Iarray);

void rmatFree(int,Rmatrix);

void cmatFree(int,Cmatrix);

void CCSFree(CCSmat);

void ReleaseManyBodyDataPkg (ManyBodyPkg);

void ReleaseEqDataPkg (EqDataPkg);

#endif

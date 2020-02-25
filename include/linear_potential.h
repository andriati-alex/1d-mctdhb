
#ifndef _linear_potential_h
#define _linear_potential_h

#include <string.h>
#include <stdio.h>
#include "array_operations.h"

void harmonic(int , Rarray , Rarray , double);

void doublewell(int , Rarray , Rarray , double , double);

void harmonicgauss(int , Rarray , Rarray , double , double , double);

void deltabarrier(int , Rarray , Rarray , double);

void barrier(int , Rarray , Rarray , double , double );

void GetPotential(int , char [], Rarray , Rarray , double , double , double );

#endif

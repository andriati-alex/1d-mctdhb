#ifndef NEW_IMAGINT_H
#define NEW_IMAGINT_H

#include "auxIntegration.h"
#include "coeffIntegration.h"
#include "configurationalSpace.h"
#include "dataStructures.h"
#include "inout.h"
#include "linearPartIntegration.h"
#include "linear_potential.h"
#include "observables.h"
#include "orbTimeDerivativeDVR.h"

int
new_imagint(EqDataPkg, ManyBodyPkg, double, int, int);

int
new_multistep_int(EqDataPkg, ManyBodyPkg, double, int, int);

#endif

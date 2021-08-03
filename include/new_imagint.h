#ifndef NEW_IMAGINT_H
#define NEW_IMAGINT_H

#include "linear_potential.h"
#include "dataStructures.h"
#include "observables.h"
#include "configurationalSpace.h"
#include "inout.h"
#include "auxIntegration.h"
#include "coeffIntegration.h"
#include "linearPartIntegration.h"
#include "orbTimeDerivativeDVR.h"

int
new_imagint(EqDataPkg, ManyBodyPkg, double, int, int);

#endif

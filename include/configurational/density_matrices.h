#ifndef DENSITY_MATRICES_H
#define DENSITY_MATRICES_H

#include "mctdhb_types.h"

/** \brief Set one-body density matrices elements */
void
set_onebody_dm(MultiConfiguration multiconf, Carray coef, Cmatrix rho);

/** \brief Set two-body density matrices elements */
void
set_twobody_dm(MultiConfiguration multiconf, Carray coef, Carray rho);

#endif

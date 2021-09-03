#ifndef DOMAIN_RESIZE_H
#define DOMAIN_RESIZE_H

#include "mctdhb_types.h"

/** \brief Reduce domain adjusting orbitals with interpolation */
void
domain_reduction(OrbitalEquation eq_desc, ManyBodyState state);

/** \brief Extent domain(trapped systems) with zeros in orbitals */
void
domain_extention(OrbitalEquation eq_desc, ManyBodyState state);

#endif

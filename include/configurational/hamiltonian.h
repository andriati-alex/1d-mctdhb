#ifndef CONFIGURATIONAL_HAMILTONIAN_H
#define CONFIGURATIONAL_HAMILTONIAN_H

#include "mctdhb_types.h"

/** \brief Evaluate action of hamiltonian and set in output array */
void
apply_hamiltonian(
    MultiConfiguration multiconf,
    Carray             coef,
    Cmatrix            hob,
    Carray             hint,
    Carray             hc);

#endif

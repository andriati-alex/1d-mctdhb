#ifndef CONFIGURATIONAL_HAMILTONIAN_H
#define CONFIGURATIONAL_HAMILTONIAN_H

#include "mctdhb_types.h"

void
apply_hamiltonian(
    MultiConfiguration multiconf,
    Carray             coef,
    Cmatrix            hob,
    Carray             hint,
    Carray             hc);

#endif

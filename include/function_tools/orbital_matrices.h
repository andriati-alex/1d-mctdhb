/** \file orbital_matrices.h
 * \author Alex Andriati
 * \date 2021/September
 * \brief Matrices involving overlap integrals of orbitals set
 *
 * Several quantities in many-body quantum dynamics require the evaluation
 * of scalar products (aka overlap) of orbitals according to the action of
 * integro-differential operators over them. Only main routines needed for
 * time integration of MCTDHB equation are defined here.
 *
 */

#ifndef ORBITAL_MATRICES_H
#define ORBITAL_MATRICES_H

#include "mctdhb_types.h"

/** \brief Set overlap matrix <Oi, Oj> */
void
set_overlap_matrix(
    OrbitalEquation eq_desc, uint16_t norb, Cmatrix orb, Cmatrix overlap);

/** \brief Set one-body orbital hamiltonian matrix */
void
set_orbital_hob(
    OrbitalEquation eq_desc, uint16_t norb, Cmatrix orb, Cmatrix hob);

/** \brief Set two-body orbital hamiltonian matrix */
void
set_orbital_hint(
    OrbitalEquation eq_desc, uint16_t norb, Cmatrix orb, Carray hint);

#endif

#ifndef INTEGRATOR_MONITOR_H
#define INTEGRATOR_MONITOR_H

#include "mctdhb_types.h"

/** \brief Eigenvalue equation numerical residue as maximum norm
 *
 * The maximum norm of a vector is `|| v || = max( abs(vi) )` for `i`
 * running from the first to the last vector index
 *
 * \param[in] multiconf multiconfigurational space descriptor
 * \param[in] coef coefficients of many-body eigenstate expansion
 * \param[in] hob orbital matrix of one-body hamiltonian
 * \param[in] hint orbital matrix of two-body interacting hamiltonian
 * \param[in] energy eigenvalue associated to `coef`
 * \return Max norm of the difference of applying the hamiltonian and
 *         multiplying by `energy` the input `coef`
 */
double
eig_residual(
    MultiConfiguration multiconf,
    Carray             coef,
    Cmatrix            hob,
    Carray             hint,
    double             energy);

/** \brief Sum of absolute values of off-diagonal overlap matrix elements
 *
 * The overlap matrix is defined from all pair-wise scalar products among
 * the orbitals, and it should be the identity for any time. This routine
 * inform how numerical errors are affecting this property.
 *
 * \param[in] norb number of orbitals
 * \param[in] grid_size number of grid points
 * \param[in] dx grid step size
 * \param[in] orb orbitals organized in matrix rows
 * \return sum of absolute values of off-diagonal elements of overlap matrix
 */
double
overlap_residual(uint16_t norb, uint16_t grid_size, double dx, Cmatrix orb);

/** \brief Trace of overlap matrix divided by number of orbitals */
double
avg_orbitals_norm(uint16_t norb, uint16_t grid_size, double dx, Cmatrix orb);

dcomplex
total_energy(ManyBodyState psi);

dcomplex
kinect_energy(OrbitalEquation eq_desc, ManyBodyState psi);

dcomplex
onebody_potential_energy(OrbitalEquation eq_desc, ManyBodyState psi);

dcomplex
interacting_energy(ManyBodyState psi);

dcomplex
virial_harmonic_residue(OrbitalEquation eq_desc, ManyBodyState psi);

double
mean_quadratic_pos(OrbitalEquation eq_desc, ManyBodyState psi);

#endif

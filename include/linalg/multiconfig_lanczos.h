#ifndef MULTICONFIG_LANCZOS_H
#define MULTICONFIG_LANCZOS_H

#include "mctdhb_types.h"

/** \brief Apply Lanczos iterative decomposition for hamiltonian
 * 
 * Use the routine to apply many-particle Hamiltonian to build the
 * Lanczos vectors, and two vectors with diagonal and off-diagonal
 * elements of the tridiagonal decomposition
 * 
 * It is an implementation with full re-orthogonalization, to minimize
 * effect of numerical errors in floating point arithmetic, and avoid
 * loss of orthogonality among eigenvectors. For more information see:
 * 
 * "Lectures on solving large scale eigenvalue problem", Peter Arbenz,
 * ETH Zurich, 2006. url : http://people.inf.ethz.ch/arbenz/ewp/
 * 
 * and other references there mentioned.
 *
 * \param[in] multiconf Multiconfigurational space type
 * \param[in] hob_mat one-body orbital-hamiltonian matrix
 * \param[in] hint_mat two-body orbital-hamiltonian matrix (interaction)
 * \param[in] iter number of iterations to evaluate
 * \param[out] diag vector of size `iter` with diagonal of decomposition
 * \param[out] offd vector of size `iter` with off-diagonal of decomposition
 * \param[out] lvec matrix with lanczos vectors in rows
 *
 * \return number of iterations (equal to `iter` if all goes well)
 */
int
lanczos(
    MultiConfiguration multiconf,
    Cmatrix            hob_mat,
    Carray             hint_mat,
    int                iter,
    Carray             diag,
    Carray             offd,
    Cmatrix            lvec);

/** \brief Compute the lowest energy coefficients vector within given orbitals
 *
 * Use the lanczos algorithm to found an approximation for the lowest energy
 * eigenstate of the configurational hamiltonian within fixed orbital basis.
 * The accuracy is subject to the number of iterations.
 *
 * \see lanczos
 *
 * \param[in] multiconf Multiconfigurational space type
 * \param[in] lanczos_work Workspace for lanczos algorithm
 * \param[in] hob_mat one-body orbital-hamiltonian matrix
 * \param[in] hint_mat two-body orbital-hamiltonian matrix (interaction)
 * \param[in/out] coef Initial guess. End with ground state approximation
 * 
 * \return Approximation for the lowest eigenvalue
 */
double
lowest_state_lanczos(
    MultiConfiguration multiconf,
    WorkspaceLanczos   lanczos_work,
    Cmatrix            hob_mat,
    Carray             hint_mat,
    Carray             coef);

#endif

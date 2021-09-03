#ifndef SPLIT_LINEAR_ORBITALS_H
#define SPLIT_LINEAR_ORBITALS_H

#include "mctdhb_types.h"
#include <mkl_dfti.h>

/** \brief Evaluate the action of linear part(one-body) orbital hamiltonian
 *
 * Use second order finite differences for derivatives
 *
 * \see linear_horb_fft
 *
 * \param[in] eq_desc equation struct descriptor
 * \param[in] orb a single orbital
 * \param[out] horb action of one-body orbital hamiltonian
 */
void
linear_horb_fd(OrbitalEquation eq_desc, Carray orb, Carray horb);

/** \brief Evaluate the action of linear part(one-body) orbital hamiltonian
 *
 * Use spectral method to evaluare derivatives
 *
 * \see linear_horb_fd
 *
 * \param[in] desc pointer to intel MKL descriptor
 * \param[in] eq_desc equation struct descriptor
 * \param[in] orb a single orbital
 * \param[out] horb action of one-body orbital hamiltonian
 */
void
linear_horb_fft(
    DFTI_DESCRIPTOR_HANDLE* fft_desc,
    OrbitalEquation         eq_desc,
    Carray                  orb,
    Carray                  horb);

/** \brief Setup the three diagonals from Crank-Nicolson method **/
void
set_cn_tridiagonal(
    OrbitalEquation eq_desc, Carray upper, Carray lower, Carray mid);

/** \brief Advance the set of orbitals according to linear part */
void advance_linear_crank_nicolson(
    OrbitalEquation, uint16_t, Carray, Carray, Carray, Cmatrix);

/** \brief Advance the set of orbitals according to linear part */
void
advance_linear_fft(DFTI_DESCRIPTOR_HANDLE*, int, int, Carray, Cmatrix);

#endif

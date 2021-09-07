#ifndef SPLIT_LINEAR_ORBITALS_H
#define SPLIT_LINEAR_ORBITALS_H

#include "mctdhb_types.h"

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
    OrbitalEquation eq_desc, OrbitalWorkspace work, Carray orb, Carray horb);

/** \brief Set exponential dvr matrix in rowmajor format */
void
set_expdvr_mat(OrbitalEquation eq_desc, Carray expdvr_mat);

/** \brief Set sine dvr matrix in rowmajor format */
void
set_sinedvr_mat(OrbitalEquation eq_desc, Carray sinedvr_mat);

/** \brief Setup the three diagonals from Crank-Nicolson method **/
void
set_cn_tridiagonal(
    OrbitalEquation eq_desc, Carray upper, Carray lower, Carray mid);

/** \brief Set exponential of derivatives part in Fourier space */
void
set_hder_fftexp(OrbitalEquation eq_desc, Carray hder_exp);

/** \brief Evaluate right-hand-side of Crank-Nicolson tridiagonal system */
void
cn_rhs(OrbitalEquation eq_desc, Carray f, Carray out);

/** \brief Advance the set of orbitals according to linear part */
void advance_linear_crank_nicolson(OrbitalEquation, OrbitalWorkspace, Cmatrix);

/** \brief Advance the set of orbitals according to linear part */
void advance_linear_fft(OrbitalWorkspace, Cmatrix);

#endif

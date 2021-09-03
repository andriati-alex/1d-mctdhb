#ifndef COEFFICIENTS_INTEGRATION_H
#define COEFFICIENTS_INTEGRATION_H

#include "mctdhb_types.h"
#include "odesys.h"

/** \brief Forwarded one time step with 2nd order Runge-Kutta */
void
propagate_coef_rk2(MCTDHBDataStruct mctdhb, Carray cnext);

/** \brief Forwarded one time step with 4th order Runge-Kutta */
void
propagate_coef_rk4(MCTDHBDataStruct mctdhb, Carray cnext);

/** \brief Forwarded one time step with 5th order Runge-Kutta */
void
propagate_coef_rk5(MCTDHBDataStruct mctdhb, Carray cnext);

/** \brief Forward one time step with Symmetric Iterative Lanczos(SIL)
 * 
 * "Unitary quantum time evolution by iterative Lanczos reduction",
 * Tae Jun Park and J.C. Light, J. Chemical Physics 85, 5870, 1986
 * https://doi.org/10.1063/1.451548
 *
 * \param[in/out] mctdhb problem descriptor struct. Update all internal
 *                       fields depending on coefficients to new time
 * \param[out] cnext coefficients forwarded to next step
 */
void
propagate_coef_sil(MCTDHBDataStruct mctdhb, Carray cnext);

#endif

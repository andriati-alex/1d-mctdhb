/** \file nonlinear_orbitals.h
 * \author Alex Andriati
 * \date 2021/September
 * \brief Nonlinear part of orbital equations routines
 *
 * These subroutines are auxiliar for time integration, evaluating the
 * nonlinear part of the coupled equations, for any orbital number and
 * grid point, including or not the nonlinearity coming from projector
 */

#include "mctdhb_types.h"

/** \brief Evaluate nonlinear part strictly from interaction
 *
 * This routine is suitable for non split-step schemes. In this case,
 * after adding linear part contribution, one must apply the projector
 *
 * \see orb_full_nonlinear for split-step treatment
 *
 * \param[in] orb_num select orbital equation number to evaluate
 * \param[in] grid_pt select a grid point between 0 and orbitals size - 1
 * \param[in] g contact interaction parameter
 * \param[in] psi Many-body state struct
 * \return value of nonlinear contribution to orbital equation `orb_num`
 *         at grid point `grid_pt`
 */
dcomplex
orb_interacting_part(int orb_num, int grid_pt, double g, ManyBodyState psi);

/** \brief Evaluate nonlinear part from interaction and projection
 *
 * Suitable for split-step schemes where this evaluate all contributions
 * for the entire nonlinear part of orbital equation.
 *
 * \see orb_interacting_part
 *
 * \param[in] orb_num select orbital equation number to evaluate
 * \param[in] grid_pt select a grid point between 0 and orbitals size - 1
 * \param[in] g contact interaction parameter
 * \param[in] psi Many-body state struct
 * \return value of nonlinear contribution to orbital equation `orb_num`
 *         at grid point `grid_pt` including action of projectors
 */
dcomplex
orb_full_nonlinear(int orb_num, int grid_pt, double g, ManyBodyState psi);

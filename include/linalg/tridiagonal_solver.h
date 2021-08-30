#ifndef TRIDIAGONAL_SOLVER_H
#define TRIDIAGONAL_SOLVER_H

#include "mctdhb_types.h"

/** \brief Solve linear system with LU factorization for tridiagonal matrix
 *
 * Used for tridiagonal systems solution from diffusion PDEs with
 * zero boundary condition
 *
 * \param[in] sys_dim dimension/size of the system
 * \param[in] upper   diagonal starting in row 1 col 2.
 *                    At least size of `sys_dim - 1`
 * \param[in] lower   diagonal starting in row 2 col 1
 *                    At least size of `sys_dim - 1`
 * \param[in] mid     Main diagonal starting at row 1 col 1
 *                    At least size of `sys_dim`
 * \param[in] rhs     Right handed side of linear system of equations
 * \param[out] sol    Array with solution of the linear system
 */
void
solve_cplx_tridiag(
    uint32_t sys_dim,
    Carray   upper,
    Carray   lower,
    Carray   mid,
    Carray   rhs,
    Carray   sol);

/** \brief Solve linear system with cyclic-tridiagonal matrix
 *
 * Used for tridiagonal systems solution from diffusion PDEs with
 * periodic boundary conditions. The periodic boundary conditions
 * implies extra two entries, in top right and bottom left of the
 * matrix.
 *
 * \note The implementation follows section 3.iii of the paper:
 * Thiab R. Taha, "Solution of Periodic Tridiagonal Linear Systems of
 * Equations on a Hypercube", Proceedings of the Fifth Distributed
 * Memory Computing Conference, 1990, Doi 10.1109/DMCC.1990.555404
 *
 * \param[in] sys_dim dimension/size of the system
 * \param[in] upper   diagonal starting in row 1 col 2. The last element must
 *                    be the top right element from the matrix. Size of
 *                    `sys_dim` is required
 * \param[in] lower   diagonal starting in row 2 col 1. The last element must
 *                    be the bottom left element from the matrix. Size of
 *                    `sys_dim` is required
 * \param[in] mid     Main diagonal starting at row 1 col 1
 *                    At least size of `sys_dim` is required
 * \param[in] rhs     Right handed side of linear system of equations
 * \param[out] sol    Array with solution of the linear system
 */
void
solve_cplx_cyclic_tridiag_lu(
    uint32_t sys_dim,
    Carray   upper,
    Carray   lower,
    Carray   mid,
    Carray   rhs,
    Carray   sol);

/** \brief Solve linear system with cyclic-tridiagonal matrix
 *
 * Just the same functionality and API of `solve_cplx_cyclic_tridiag_lu`
 * but use the Sherman-Morrison formula, which is used to solve systems
 * with small modifications from a known one.
 *
 * See Numerical Recipes book, p. ~ 70.
 *
 * \see solve_cplx_cyclic_tridiag_lu
 */
void
solve_cplx_cyclic_tridiag_sm(
    uint32_t sys_dim,
    Carray   upper,
    Carray   lower,
    Carray   mid,
    Carray   rhs,
    Carray   sol);

#endif

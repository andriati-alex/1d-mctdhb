#ifndef MCTDHB_INTEGRATOR_DRIVER_H
#define MCTDHB_INTEGRATOR_DRIVER_H

#include "mctdhb_types.h"

/** \brief Tolerance for overlap residue below which evolution is aborted */
#define REALTIME_OVERLAP_RESIDUE_TOL 0.001

/** \brief Set periodic verification for anticipated stop
 *
 * Each physical problem require specific time scales for convergence in
 * imag time propagation, and auto convergence criterias may be employed
 * to avoid 'oversolving'. This function can enable or disable automatic
 * verification for convergence in \c integration_driver routine to stop
 * if some requirements are satisfied. Functions to change these default
 * requirements are provided
 *
 * \see set_energy_convergence_digits
 * \see set_energy_convergence_eig_residual
 *
 * \param[in] must_check TRUE(1) or FALSE(0) to enable or not verification
 */
void
set_autoconvergence_check(Bool must_check);

/** \brief Min energy digits stabilization to stop imag time propagation */
void
set_energy_convergence_digits(uint8_t edig);

/** \brief Max eigenvalue residue tolerance to stop imag time propagation */
void
set_energy_convergence_eig_residual(double eig_res);

/** \brief Max overlap residue to skip orthogonality improvement
 *
 * By default the MCTDHB method assume a perfect orthogonality of the
 * orbitals used at every time step. Due to accumulated errors during
 * time propagation this property may (probably) be lost in long time
 * evolutions. To mitigate this effect, the inverse of overlap matrix
 * can be used in the projector part. This function set a threshold
 * below which the inverse of overlap matrix is not used based on the
 * overlap residual definition
 *
 * \see overlap_residual
 *
 * \param[in] over_res The threshold below which overlap inverse is not used
 */
void
set_overlap_residue_threshold(double over_res);

/** \brief Propagate one step both equations for coefficients and orbitals
 *
 * Use the integration methods set in \c mctdhb struct reference. After
 * the propagation of coefficients and orbitals it synchronizes related
 * structs, as density matrices and one- and two-particle orbital matrices
 *
 * \param[in/out] mctdhb Main struct reference to propagate in time
 * \param[in] orb_works  Orbital workspace
 * \param[in] coef_works Coefficients workspace
 * \param[in] curr_step  Current step input data refers to
 */
void
mctdhb_propagate_step(
    MCTDHBDataStruct mctdhb,
    Carray           orb_works,
    Carray           coef_works,
    uint32_t         curr_step);

/** \brief Driver routine to propagate recording data
 *
 * Auto propagate many steps until \c tend monitoring some important
 * quantities and recording(real time) data in intermediate steps or
 * the just final step(imag time).
 * 
 * \param[in] mctdhb Main struct with data to be propagated in time
 * \param[in] rec_nsteps Steps interval to record data for real time
 * \param[in] prefix Common file name prefix to use in output files
 * \param[in] tend Final time propagation (override \c mctdhb value)
 */
void
integration_driver(
    MCTDHBDataStruct mctdhb,
    uint16_t         rec_nsteps,
    char             prefix[],
    double           tend,
    uint8_t         monitor_rate);

#endif

/** \file mctdhb_types.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date 2021/August
 * \brief Define main data structures
 *
 * The data structures are meant to simplify the amount of arguments needed
 * in function calls. For that, zip a collection of data relevant for each
 * module in this package
 */

#ifndef MCTDHB_TYPES_H
#define MCTDHB_TYPES_H

#include <complex.h>
#include <inttypes.h>
#include <mkl_dfti.h>
#include <mkl_types.h>

/** \brief Double precision pi value */
#define PI 3.141592653589793
/** \brief Maximum length of stack strings */
#define STR_BUFF_SIZE 128
/** \brief Maximum grid size supported */
#define MAX_GRID_SIZE 2049
/** \brief Minimum grid size required */
#define MIN_GRID_SIZE 50
/** \brief Maximum number of single particle states(orbitals) supported */
#define MAX_ORBITALS 30
/** \brief Maximum number of single particle states(orbitals) supported */
#define MAX_PARTICLES 10000
/** \brief Maximum number of iterations for Lanczos integrator **/
#define MAX_LANCZOS_ITER 15
/** \brief Minimum number of iterations for Lanczos integrator **/
#define MIN_LANCZOS_ITER 2

typedef double complex dcomplex;
typedef int*           Iarray;
typedef int**          Imatrix;
typedef double*        Rarray;
typedef double**       Rmatrix;
typedef dcomplex*      Carray;
typedef dcomplex**     Cmatrix;
typedef MKL_Complex16* MKLCarray;

/** \brief Custom boolean type with enum */
typedef enum
{
    FALSE=0,
    TRUE=1
} Bool;

/** \brief Time integration type for ground state or dynamics calculation */
typedef enum
{
    IMAGTIME = 10, //! For ground-state convergence
    REALTIME = 11  //! For dynamics from initial condition
} IntegratorType;

/** \brief Coefficients integration type */
typedef enum
{
    LANCZOS = 20,
    RUNGEKUTTA = 21
} CoefIntegrator;

/** \brief Orbital integration type */
typedef enum
{
    FULLSTEP_RUNGEKUTTA = 30,
    SPLITSTEP = 31,
} OrbIntegrator;

/** \brief Complementary information to evaluate derivatives */
typedef enum
{
    DVR = 300,
    SPECTRAL = 301,
    FINITEDIFF = 302
} OrbDerivative;

/** \brief Global Runge-Kutta order for all methods using it */
typedef enum
{
    RK2 = 2,
    RK4 = 4,
    RK5 = 5
} RungeKuttaOrder;

/** \brief Specific for finite differences on how to set boundaries */
typedef enum
{
    ZERO_BOUNDS = 1000,
    PERIODIC_BOUNDS = 1001
} BoundaryCondition;

/** \brief Default time integration as imaginary */
#define DEFAULT_INTEGRATION_TYPE IMAGTIME
/** \brief Default integrator for coefficients as SIL */
#define DEFAULT_COEF_INTEGRATOR LANCZOS
/** \brief Default integrator for orbitals as full-step Runge-Kutta */
#define DEFAULT_ORB_INTEGRATOR FULLSTEP_RUNGEKUTTA
/** \brief Default way to handle derivatives is using DVR */
#define DEFAULT_ORB_DERIVATIVE DVR
/** \brief Default Runge-Kutta methods order */
#define DEFAULT_RUNGEKUTTA_ORDER RK5
/** \brief Default boundary conditions are periodic */
#define DEFAULT_BOUNDARY_CONDITION PERIODIC_BOUNDS
/** \brief Default number of iterations in SIL integrator */
#define DEFAULT_LANCZOS_ITER 5

/** \brief Single particle potential function signature
 *
 * This is the general signature to evaluate one-body time-dependent
 * potential in spatial grid at any time instant
 *
 * \param[in] t         time instant
 * \param[in] npts      number of points in the grid
 * \param[in] pts       array with grid points
 * \param[in] params    extra parameters defined by the client
 * \param[out] grid_pot potential values at grid points
 */
typedef void (*single_particle_pot)(
    double t, uint16_t npts, double* pts, void* params, double* grid_pot);

/** \brief Time dependent parameter function signature
 *
 * Signature to declare time-dependent parameters. Only for interaction
 *
 * \param[in] t      time instant
 * \param[in] params extra needed parameters from client
 */
typedef double (*time_dependent_parameter)(double t, void* params);

/** \brief Spatial equation setup for orbitals */
typedef struct
{
    char                     eq_name[STR_BUFF_SIZE];
    uint16_t                 norb, grid_size;
    BoundaryCondition        bounds;
    double                   t, xi, xf, dx, tstep, tend, g, d2coef;
    dcomplex                 d1coef, prop_dt, time_fac;
    Rarray                   grid_pts;
    Rarray                   pot_grid;
    void*                    pot_extra_args;
    void*                    inter_extra_args;
    single_particle_pot      pot_func;
    time_dependent_parameter inter_param;
} _OrbitalEquation;

typedef _OrbitalEquation* OrbitalEquation;

/** \brief Mappings to track action of creation/destruction operators */
typedef struct
{
    uint32_t *map, *mapot, *maptt, *strideot, *stridett;
} _OperatorMappings;

typedef _OperatorMappings* OperatorMappings;

/** \brief Multiconfigurational space setup */
typedef struct
{
    uint16_t         npar, norb;
    uint32_t         dim;
    uint16_t*        hash_table;
    uint32_t*        subspaces_dim;
    OperatorMappings op_maps;
} _MultiConfiguration;

/** \brief Reference to \c _MultiConfiguration */
typedef _MultiConfiguration* MultiConfiguration;

/** \brief Many-body state in Multiconfigurational problem */
typedef struct
{
    uint16_t npar, norb, grid_size;
    uint32_t space_dim;
    Carray   coef, hint, tb_denmat;
    Cmatrix  hob, orbitals, ob_denmat, inv_ob_denmat;
} _ManyBodyState;

/** \brief Reference to \c _ManyBodyState */
typedef _ManyBodyState* ManyBodyState;

/** \brief Memory struct needed to evaluate Lanczos iteratios */
typedef struct
{
    uint16_t iter;
    uint32_t space_dim;
    Cmatrix  lanczos_vectors;
    Carray   decomp_diag, decomp_offd, coef_lspace, transform, hc, reortho;
    Rarray   lapack_diag, lapack_offd, lapack_eigvec;
} _WorkspaceLanczos;

/** \brief Reference to \c _WorkspaceLanczos */
typedef _WorkspaceLanczos* WorkspaceLanczos;

/** \brief Memory workspace for orbital integration */
typedef struct
{
    uint16_t               norb, grid_size;
    OrbDerivative          orb_der_method;
    Bool                   impr_ortho;
    Carray                 dvr_mat;
    Carray                 cn_upper, cn_lower, cn_mid, cn_rhs;
    Cmatrix                orb_work1, orb_work2;
    void*                  extern_work;
    DFTI_DESCRIPTOR_HANDLE fft_desc;
    Rarray                 fft_freq;
    Carray                 fft_hder_exp;
} _OrbitalWorkspace;

/** \brief Reference to \c _OrbitalEquation */
typedef _OrbitalWorkspace* OrbitalWorkspace;

/** \brief Memory workspace for coefficients integration */
typedef struct
{
    WorkspaceLanczos lan_work;
    void*            extern_work;
} _CoefWorkspace;

/** \brief Reference to \c _CoefWorkspace */
typedef _CoefWorkspace* CoefWorkspace;

/** \brief Master struct with all information for MCTDHB numerical problem */
typedef struct
{
    IntegratorType     integ_type;
    CoefIntegrator     coef_integ_method;
    OrbIntegrator      orb_integ_method;
    OrbDerivative      orb_der_method;
    RungeKuttaOrder    rk_order;
    MultiConfiguration multiconfig_space;
    OrbitalEquation    orb_eq;
    CoefWorkspace      coef_workspace;
    OrbitalWorkspace   orb_workspace;
    ManyBodyState      state;
} _MCTDHBDataStruct;

/** \brief Reference to \c _MCTDHBDataStruct */
typedef _MCTDHBDataStruct* MCTDHBDataStruct;

#endif

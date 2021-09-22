/** \file dataio.h
 * \author Alex Andriati
 * \date September/2021
 * \brief Input and output with text files
 *
 * Due to the general approach from the API developed in this package, even
 * a simple MCTDHB problem can require a formidable labor to setup. The API
 * provided in this file ease the data input and output with with text files.
 * Moreover, abstract some concepts of multiple jobs run, resulting in very
 * general API capable to handle parameters sweep, possibly within analytic
 * continuation, feeding next job input with last one's output.
 *
 * Generally, to keep the runs organized, two directories to keep input and
 * output files are required. The `inp_dirname` and `out_dirname` global var
 * set the default directory names. Other global vars set complex numbers
 * format in text files (usually python compatible) and them can be changed
 * from a client application if needed.
 *
 * The general workflow in recording data is to give a prefix (recommended
 * to be related to input prefix data) and the output directory is set by
 * the global variable \c out_dirname. The complete file names with suffix
 * customized according to data being save is achieved with \c set_output_fname
 */

#ifndef MCTDHB_DATAIO_H
#define MCTDHB_DATAIO_H

#include "mctdhb_types.h"

/** \brief Formatter sequence for function definition with real parameters */
#define BUILTIN_TIME_FUNCTION_INPUT_FMT "%s %lf %lf %lf %lf %lf"
/** \brief Minimum values per line in input parameters files */
#define MIN_PARAMS_INLINE 21
/** \brief Default file suffix to read parameters */
#define PARAMS_FNAME_SUFFIX "_mctdhb_parameters.dat"

/** \brief Complex number formatter for orbitals (default python matrix) */
extern char orb_cplx_read_fmt[STR_BUFF_SIZE];
/** \brief Complex number formatter for coefficients (default python array) */
extern char coef_cplx_read_fmt[STR_BUFF_SIZE];
/** \brief Default input directory name */
extern char inp_dirname[STR_BUFF_SIZE];
/** \brief Default output directory name */
extern char out_dirname[STR_BUFF_SIZE];
/** \brief Default integrator descriptor file name */
extern char integrator_desc_fname[STR_BUFF_SIZE];

/** \brief Control number of decimal digits to display energy */
extern uint8_t monitor_energy_digits;
/** \brief Whether to display minimum occupation or not */
extern Bool monitor_disp_min_occ;
/** \brief Whether to display kinect energy or not */
extern Bool monitor_disp_kin_energy;
/** \brief Whether to display interacting energy or not */
extern Bool monitor_disp_int_energy;
/** \brief Whether to display orbitals overlap residue or not */
extern Bool monitor_disp_overlap_residue;
/** \brief Whether to display orbitals (average) norm or not */
extern Bool monitor_disp_orb_norm;
/** \brief Whether to display coefficients norm or not */
extern Bool monitor_disp_coef_norm;
/** \brief Whether to display hamiltonian eigvalue residue or not */
extern Bool monitor_disp_eig_residue;

/** \brief Multiple jobs input handling */
typedef enum
{
    COMMON_INP,   //! Always use job 1
    MULTIPLE_INP, //! Use job from input param
    LAST_JOB_OUT  //! Use from last output (requies same prefix convention)
} JobsInputHandle;

/** \brief Available data to record as output */
typedef enum
{
    ORBITALS_REC,
    COEFFICIENTS_REC,
    ONE_BODY_MATRIX_REC,
    TWO_BODY_MATRIX_REC,
    PARAMETERS_REC,
    ONE_BODY_POTENTIAL_REC
} RecordDataType;

/** \brief Read and set orbitals(grid along rows and orb number along columns)
 *
 * This function is complementary to \c get_mctdhb_from_files to set
 * orbitals of sub-struct \c ManyBodyState in \c MCTDHBDataStruct
 *
 * \warning The \c orbital_matrices are not set automatically.
 *          Use the functions provided in \c orbital_matrices.h
 *          to complete set orbital dependencies
 *
 * \see orbital_matrices.h
 * \see ManyBodyState
 *
 * \param[in] fname file name(full path) with data for orbitals
 * \param[in] psi   Many-Body state struct reference to record values
 */
void
set_orbitals_from_file(char fname[], ManyBodyState psi);

/** \brief Read coefficients as column vector from file
 *
 * This function is complementary to \c get_mctdhb_from_files to set
 * coefficients of sub-struct \c ManyBodyState in \c MCTDHBDataStruct
 *
 * \warning The density matrices are not set automatically.
 *          Use the functions in \c density_matrices.h to
 *          complete set coefficients dependencies
 *
 * \see density_matrices.h
 * \see ManyBodyState
 *
 * \param[in] fname file name(full path) with coef as column matrix
 * \param[in] psi   Many-Body state struct reference to record values
 */
void
set_coef_from_file(char fname[], ManyBodyState psi);

/** \brief Get MCTDHB data type provided information from files
 *
 * There are several parameters to tweak to fully set a (numerical)MCTDHB
 * problem, but some of them are required to at least allocate all needed
 * memory and set the configurational space, such as hashing and operator
 * mappings. This function uses a line of a file to allocate all necessary
 * memory and set basic structs/fields(as multiconf space). The parameters
 * must be disposed in a single line in the following order
 *
 * \li \c npar
 * \li \c norb
 * \li \c grid_size
 * \li \c xi
 * \li \c xf
 * \li \c tstep
 * \li \c tend
 * \li \c d2_coef
 * \li \c imagpart_d1coef
 * \li \c g_func_name
 * \li sequence of 5 real values
 * \li \c pot_func_name
 * \li sequence of 5 real values
 *
 * Most part of the parameters above are self-explanatory, but there are two
 * strings `g_func_name` and `pot_func_name` which refers to names of some
 * of the builtin time-dependent functions for the contact interaction and
 * the single particle potential. Available names are provided in headers
 * `builtin_potential.h` and `builtin_time_parameter.h` which includes some
 * known examples. If using the main executable, the client must choose a
 * builtin case, but if a separate executable is calling this function the
 * "custom" string can be used to indicate that a custom potential will be
 * given as pointer to a function (following the specific signature). The
 * parameters given in the file are ignored if custom parameters are given
 *
 * \warning Only space and integration related internal structs are set.
 *          Values in \c ManyBodyState sub-struct still must be set. Use
 *          \c set_?_from_file routines to set initial data.
 *
 * \see set_orbitals_from_file
 * \see set_coef_from_file
 *
 * \param[in] par_fname           full path to file with parameters
 * \param[in] integ_fname         (OPTIONAL) full path to file with integrator
 *                                descriptor
 * \param[in] line                line number to read
 * \param[in] custom_pot_fun      client provided (trap) potential to be used
 *                                in case the string read in file is "custom"
 *                                (ignored if string "custom" is not in file)
 * \param[in] custom_inter_fun    client provided time dependent function to
 *                                interaction parameter if file is "custom"
 *                                (ignored if string "custom" is not in file)
 * \param[in] custom_pot_params   overwrite the file parameters to evaluate
 *                                (trap) potential.
 * \param[in] custom_inter_params overwrite the file paramters to evaluate
 *                                time-dependent interaction strength
 *
 * \return MCTDHB struct reference with all memory allocated, integrator
 *                descriptor and multi config space set. All the initial
 *                data for coefficients, orbitals and their matrices are
 *                NOT set.
 */
MCTDHBDataStruct
get_mctdhb_from_files(
    char                     par_fname[],
    char                     integ_fname[],
    uint32_t                 line,
    single_particle_pot      custom_pot_fun,
    time_dependent_parameter custom_inter_fun,
    void*                    custom_pot_params,
    void*                    custom_inter_params);

/** \brief Complete definition of MCTDHB problem using execution directory
 *
 * Essentially, use the functions `get_mctdhb_from_files` to create all
 * memory workspace and set integration method. Besides set values for
 * orbitals and coefficients including their related matrices.
 *
 * The directory specified by global variable \c inp_dirname
 * (default "./input") must contain:
 * \li \c fprefix_mctdhb_parameters.dat file used in `get_mctdhb_from_files`
 * \li \c fprefix_job?_orb.dat file with orbitals as matrix (grid pts as rows)
 * \li \c fprefix_job?_coef.dat file with coefficients vector (column matrix)
 *
 * with \c fprefix given in this function arg and '?' is a number depending
 * on the job given by \c job_num and job handler \c which_inp . The way it
 * reads behave as:
 * \li \c COMMON_INP is equivalent to set `? == 1` in the files independently
 *     of \c job_num specified it will use the same input.
 * \li \c MULTIPLE_INP will set \c ? to the \c job_num given
 * \li \c LAST_JOB_OUT will try to set using output files directory from last
 *     job evaluated. This requires a sequential execution of jobs, and the
 *     same naming convention for output files with given \c prefix
 *
 * In current directory \c "./" must contain the integrator descriptor file
 * with the name in the string global variable \c integrator_desc_fname and
 * as default is \c "mctdhb_integrator.conf" . A client app can change this
 * value, but this is used by the main executable. This is optional and case
 * the file is not found use default values. See the macros \c DEFAULT_? in
 * \c mctdhb_types.h file
 *
 * \see PARAMS_FNAME_SUFFIX
 * \see mctdhb_types.h
 * \see get_mctdhb_from_files
 *
 * \param[in] fprefix             file names common prefix in input directory
 * \param[in] job_num             job number (the line read in parameters)
 * \param[in] which_inp           How to handle input for multiple jobs
 * \param[in] custom_pot_fun      client custom (trap) potential function
 * \param[in] custom_inter_fun    client custom time-dependent interaction
 * \param[in] custom_pot_params   overwrite file parameters for potential
 * \param[in] custom_inter_params overwrite file paramters for interaction
 */
MCTDHBDataStruct
full_setup_mctdhb_current_dir(
    char                     fprefix[],
    uint32_t                 job_num,
    JobsInputHandle          which_inp,
    single_particle_pot      custom_pot_fun,
    time_dependent_parameter custom_inter_fun,
    void*                    custom_pot_params,
    void*                    custom_inter_params);

/** \brief Print on screen a nice banner with project name */
void
screen_display_banner();

/** \brief automatically detect number of jobs by lines in params file
 *
 * The params file must be in directory \c inp_dirname (default \c "./input")
 * and forced to have as suffix the string \c "_mctdhb_parameters.dat" given
 * in the macro \c PARAMS_FNAME_SUFFIX
 *
 * \see PARAMS_FNAME_SUFFIX
 *
 * \param[in] prefix file name part preceeding \c "_mctdhb_parameters.dat"
 *
 * \return number of lines in parameters file to be used as number of jobs
 */
uint32_t
auto_number_of_jobs(char prefix[]);

/** \brief Display on screen general info about the problem setup */
void
screen_display_mctdhb_info(
    MCTDHBDataStruct mctdhb, Bool disp_integ, Bool disp_mem, Bool disp_monitor);

/** \brief Set file name to write output data
 *
 * Set output file name depending on data to be recorded using common prefix.
 * It is recommended to use the same prefix from input files (if provided) as
 * the identification becomes easier. Moreover, \c prefix should provide info
 * about time integration type and a job number in case the execution handles
 * multiple jobs. The directory (path) name to record data is in \c out_dirname
 * global variable and defaults to \c ./output
 *
 * \param[in] prefix file names prefix to use
 * \param[in] rec_id Data type identification to be recorded
 * \param[out] fname final file name setup
 */
void
set_output_fname(char prefix[], RecordDataType rec_id, char* fname);

/** \brief Display columns being printed during time integration
 *
 * These columns are defined by the set of global variables starting
 * with \c monitor_disp prefix which are all boolean to select which
 * information shall be printed
 */
void
screen_integration_monitor_columns();

/** \brief Show on screen key quantities during time propagation */
void
screen_integration_monitor(MCTDHBDataStruct mctdhb);

/** \brief Record custom set of data fields to represent the many-body state
 *
 * Instead of using the coefficients record the one- and two-body
 * density matrices, which are sufficient to compute most part of
 * physical observables. This pre-processing layer is advantageous
 * since the density matrices must be computed anyway and make it
 * easier to analyze data in any other language. All are recorded
 * appending to files as inline array, so matrices are translated
 * to rowmajor format. This is suitable for realtime propagations
 * with each line corresponding to a time step.
 *
 * For example, the orbitals are sequentially recorded resulting
 * in inline array with size `norb * grid_size`
 *
 * The full filename is defined using the the `prefix` param with
 * suffix convention for each file given in \c set_output_fname .
 * The output dir can be controlled through the global variable
 * \c out_dirname which default is \c ./output
 *
 * \note The files are open in append mode so repeting the same
 *       prefix will NOT overwrite data, but continue from last line
 *
 * \note This shall be prefered for real time calculations to reduce
 *       post computational cost and eventually storage requirements
 *
 * \see set_output_fname
 * \see record_raw_state
 *
 * \param[in] prefix file names prefix (den matrices and orbitals)
 * \param[in] psi    Many-Body state struct reference
 */
void
append_processed_state(char prefix[], ManyBodyState psi);

/** \brief Record only Fock-basis expansion coefficients and orbitals
 *
 * Record minimal information to represent many-body state through
 * coefficients and orbitals. Differently of \c append_processed_state
 * the values are not appended and data is overwritten in case the file
 * already exists. The file names are decided according to \c set_output_fname
 * to set the suffix with the \c prefix given
 *
 * \see set_output_fname
 * \see append_processed_state
 *
 * \li Coefficients of Fock basis expansion recorded as column matrix
 * \li Orbitals recorded with grid points along rows
 *
 * \param[in] prefix Common file names prefix
 * \param[in] psi    Many-Body state struct reference
 */
void
record_raw_state(char prefix[], ManyBodyState psi);

/** \brief Record array with values of contact interaction for all time steps
 *
 * The array is recorded as column matrix with 10 decimal places.
 * The file name suffix is fixed to \c "_interaction.dat" in the
 * directory set from global variable \c out_dirname which default
 * is \c "./output"
 *
 * \note This function must be called once and the values are recorded
 *       for all time steps
 *
 * \param[in] prefix  file name prefix
 * \param[in] eq_desc equation descriptor struct
 */
void
record_time_interaction(char prefix[], OrbitalEquation eq_desc);

/** \brief Append/Record current one-body potential used
 *
 * Use the time in \c OrbitalEquation struct to evaluate and record
 * potential as inline array. If the file already exists the data is
 * appended in the last line. The file name suffix is fixed to
 * \c "_obpotential.dat" recorded in the directory set from global
 * variable \c out_dirname which default is \c "./output"
 *
 * \param[in] prefix  file name prefix
 * \param[in] eq_desc equation descriptor struct (\c eq_desc->pot_grid is used)
 */
void
append_timestep_potential(char prefix[], OrbitalEquation eq_desc);

/** \brief Record data array with time step values
 *
 * The directory is set from global variable \c out_dirname which default
 * is \c "./output" and the file name suffix is fixed as \c "_timesteps.dat"
 *
 * \param[in] prefix file name prefix to use
 * \param[in] tend   last time step value
 * \param[in] tstep  time step size
 */
void
record_time_array(char prefix[], double tend, double tstep);

/** \brief Record main parameters to setup the problem
 *
 * This is the analogous function of \c get_mctdhb_from_files but to write
 * main equation parameters to help post analysis in any other platform. A
 * file in \c out_dirname is open in append mode, with the \c prefix given
 * added by the suffix "_mctdhb_paramters.dat". The one-body potential and
 * interaction function names are not recorded, instead just the value in
 * \c mctdhb->orb_eq->g is recorded (see list below). For the potential use
 * the cpydataio lib or the function \c append_timestep_potential
 *
 * The parameters are written in the following order
 *
 * \li \c npar
 * \li \c norb
 * \li \c grid_size
 * \li \c xi
 * \li \c xf
 * \li \c time_step
 * \li \c finale_time
 * \li \c d2coef
 * \li \c d1coef.imag
 * \li \c mctdhb->orb_eq->g
 * \li energy of \c mctdhb->state
 *
 * and a line break to easy recording multiple jobs in each line
 *
 * \see get_mctdhb_from_files
 * \see append_timestep_potential
 */
void
record_mctdhb_parameters(char prefix[], MCTDHBDataStruct mctdhb);

#endif

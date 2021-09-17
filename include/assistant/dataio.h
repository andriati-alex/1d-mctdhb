/** \file dataio.h
 * \author Alex Andriati
 * \date September/2021
 * \brief Input and output with text files
 *
 * Due to the general approach from the API developed in this package, even
 * a simple problem MCTDHB problem can provide some labor to setup. The API
 * provided in this file ease the data input and output with with text files.
 * Moreover, abstract some concepts of multiple jobs run, resulting in very
 * general API capable to handle parameters sweep, possibly within analytic
 * continuation feeding next job input with last job output.
 *
 * Generally, to keep the runs organized, two directories to keep input and
 * output files is required. The `inp_dirname` and `out_dirname` global var
 * set the default directory names. Other global vars set complex numbers
 * format in text files (usually python compatible) and some other helper
 * parameters available to the client to customize.
 */
#ifndef MCTDHB_DATAIO_H
#define MCTDHB_DATAIO_H

#include "mctdhb_types.h"

/** \brief Formatter sequence for function definition with real parameters */
#define BUILTIN_TIME_FUNCTION_INPUT_FMT "%s %lf %lf %lf %lf %lf"
/** \brief Minimum parameters per line in input params files */
#define MIN_PARAMS_LINE 21
/** \brief Default complex number reading formatter for orbitals */
extern char orb_cplx_read_fmt[STR_BUFF_SIZE];
/** \brief Default complex number reading formatter for coefficients */
extern char coef_cplx_read_fmt[STR_BUFF_SIZE];
/** \brief Default input directory name */
extern char inp_dirname[STR_BUFF_SIZE];
/** \brief Default output directory name */
extern char out_dirname[STR_BUFF_SIZE];
/** \brief Default integrator descriptor file name */
extern char integrator_desc_fname[STR_BUFF_SIZE];

/** \brief Multiple jobs input handling */
typedef enum
{
    COMMON_INP,
    MULTIPLE_INP,
    LAST_JOB_OUT
} JobsInputHandle;

/** \brief Available datatypes to record as output */
typedef enum
{
    ORBITALS_REC,
    COEFFICIENTS_REC,
    ONE_BODY_MATRIX_REC,
    TWO_BODY_MATRIX_REC,
    PARAMETERS_REC
} RecordDataType;

/** \brief Control verbosity in integration drivers */
typedef enum
{
    MINIMUM_VERB,
    MEDIUM_VERB,
    MAXIMUM_VERB
} Verbosity;

/** \brief Read orbitals (grid pts as rows) */
void
set_orbitals_from_file(char fname[], ManyBodyState psi);

/** \brief Read coefficients as column vector */
void
set_coef_from_file(char fname[], uint32_t space_dim, ManyBodyState psi);

/** \brief Get MCTDHB problem set provided minimal information from file
 *
 * There are several parameters to tweak to fully set a (numerical) MCTDHB
 * problem, but some of them are required to at least allocate all needed
 * memory and set the configurational space, such as hashing and operators
 * mappings. This function uses a line of a file to allocate all necessary
 * memory and set basic structs/fields (as multiconf space). Requires the
 * following values in this specific order for a given file line:
 *
 * \code
 * npar norb grid_size xi xf tstep tend d2_coef imagpart_d1coef g_func_name
 * sequence_5params pot_func_name sequence_5params
 * \endcode
 *
 * Most part of the parameters above are self-explained, but there are two
 * strings `g_func_name` and `pot_func_name` which refers to names of some
 * of the builtin time-dependent functions for the (contact)interaction
 * and single particle potential. Available names are provided in headers
 * `builtin_potential` and `builtin_time_parameter` or "custom" string can
 * be provided to use a client implementation instead. Each builtin case
 * demands up to five real parameters and those must be provided after the
 * name of the function
 *
 * \param[in] fname               full path to file to read
 * \param[in] line                line number to read
 * \param[in] custom_pot_fun      client provided (trap) potential to be used
 *                                in case the string read in file is "custom"
 * \param[in] custom_inter_fun    client provided time dependent function to
 *                                interaction parameter if file is "custom"
 * \param[in] custom_pot_params   overwrite the file parameters to evaluate
 *                                (trap) potential
 * \param[in] custom_inter_params overwrite the file paramters to evaluate
 *                                time-dependent interaction parameter
 * 
 * \return MCTDHB struct reference with all memory allocated and default
 *         integration configuration. No data is set for orbitals neither
 *         for coefficients.
 */
MCTDHBDataStruct
get_mctdhb_datafiles(
    char                     par_fname[],
    char                     integ_fname[],
    uint32_t                 line,
    single_particle_pot      custom_pot_fun,
    time_dependent_parameter custom_inter_fun,
    void*                    custom_pot_params,
    void*                    custom_inter_params);

/** \brief Complete definition of MCTDHB problem using execution directory
 *
 * Essentially use the functions `get_mctdhb_datafile_line` to create all
 * memory workspace and `set_mctdhb_integrator_from_file` to define the
 * integration method. Besides set values for orbitals and coefficients
 * using extra input files, selecting a specific job identifier which is
 * also the line number to read in `get_mctdhb_datafile_line`. All files
 * needed must share a commom prefix and be located in specific directories
 *
 * In directory "./input" must contain:
 * \li \c fprefix_mctdhb_parameters.dat file used in `get_mctdhb_datafile_line`
 * \li \c fprefix_job?_orb.dat file with orbitals as matrix (grid pts as rows)
 * \li \c fprefix_job?_coef.dat file with coefficients vector (column matrix)
 *
 * where `fprefix` is the input string parameter and '?' is a number depending
 * on the job identifier `job_num` and job handler `which_inp`.
 *
 * In current directory "./" must contain:
 * \li \c integrator_desc_fname which by default is "mctdhb_integrator.conf".
 *        The client can change using this global variable and is not realted
 *        with \c fprefix
 * 
 * \note If \c which_inp is \c LAST_JOB_OUT then files must also be provided
 *       in directory \c ./output but this case is reserved for analytical
 *       continuation for multiple jobs, where several jobs are executed
 *       sequentially with the output files sharing the same prefix with
 *       the one used here. In this case, the result of last job is used
 *       as input for the next and so on, which is useful when varying a
 *       parameter of the problem. This shall not work if either number of
 *       orbitals or particles change.
 * 
 * \param[in] fprefix             file names common prefix
 * \param[in] job_num             job number identifier
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
 * multiple jobs. The directory (path) name to record data is in `out_dirname`
 * global variable and defaults to \c ./output
 *
 * \param[in] prefix file names prefix to use
 * \param[in] rec_id Data type identification to be recorded
 * \param[out] fname final file name setup
 */
void
set_output_fname(char prefix[], RecordDataType rec_id, char* fname);

/** \brief Show on screen key quantities during time propagation */
void
screen_integration_monitor(MCTDHBDataStruct mctdhb, Verbosity verb);

/** \brief Record set of data fields representing the many-body state */
void
record_custom_data_selection(char prefix[], ManyBodyState psi);

/** \brief Record only Fock-basis expansion coefficients and orbitals */
void
record_raw_data(char prefix[], ManyBodyState psi);

void
record_time_interaction(char prefix[], OrbitalEquation eq_desc);

void
append_timestep_potential(char prefix[], OrbitalEquation eq_desc);

void
record_time_array(char prefix[], double tend, double tstep);

#endif

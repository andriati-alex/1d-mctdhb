/** \file space.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date September/2021
 * \brief Module for multiconfigurational space handling
 */

#ifndef CONFIGURATIONAL_SPACE_H
#define CONFIGURATIONAL_SPACE_H

#include "mctdhb_types.h"

/** \brief return n! */
uint64_t
fac(uint8_t n);

/** \brief Exit with error if parameters are invalid */
void
assert_space_parameters(uint16_t npar, uint16_t norb);

/** \brief Dimension of multiconfigurational Hilbert space */
uint32_t
space_dimension(uint16_t npar, uint16_t norb);

/** \brief Auxiliar array for subspaces dimension storage
 *
 * Set array with all results of space dimension for every number of
 * particles and orbitals less than or equal to `npar` and `norb`
 * This is used in routines that require conversion of configuration
 * to its hash index, that is, depends on `hash_index` routine.
 * \see hash_index
 *
 * \param[in] npar number of particles
 * \param[in] norb number of orbitals
 */
uint32_t*
get_subspaces_dim(uint16_t npar, uint16_t norb);

/** \brief Return array with all configurations sorted by their hash index */
uint16_t*
get_hash_table(uint16_t npar, uint16_t norb);

/** \brief Set occupation numbers in a configuration from its hash index
 *
 * The inverse operation of `hash_index`
 * \see hash_index
 *
 * \param[in] hash_i hashing index to set the configuration
 * \param[in] npar number of particles
 * \param[in] norb number of orbitals
 * \param[out] conf configuration to compute its hashing index
 */
void
set_config(uint32_t hash_i, uint16_t npar, uint16_t norb, uint16_t* conf);

/** \brief Get the hash index of a configuration
 *
 * The inverse operation of `set_config`. Due to intense use in other
 * routines requires `subspaces_dim` for better performance
 *
 * \see set_config
 * \see get_subspaces_dim
 *
 * \param[in] npar number of particles
 * \param[in] norb number of orbitals
 * \param[in] subspaces_dim array with dimensions of all subspaces
 * \param[in] conf configuration to compute its hashing index
 * \return hashing index of the input configuration
 */
uint32_t
hash_index(
    uint16_t npar, uint16_t norb, uint32_t* subspaces_dim, uint16_t* conf);

/** \brief Mapping for action of pair of creation destruction operator
 *
 * Given any first configuration index, map to a second one obtained
 * by replacing one particle. Thus given the first index `i`, we have
 * `Map[i + k * nc + l * nc * M] =` index of configuration which have
 * one particle less in `k` that has been placed in `l`
 *
 * \param[in] npar       number of particles
 * \param[in] norb       number of orbitals
 * \param[in] subsdim    subspaces dimensions from `get_subspaces_dim`
 * \param[in] hash_table configurational space hashing table
 * \return Array with mappings. See how to use in the description above
 */
uint32_t*
get_single_jump_map(
    uint16_t npar, uint16_t norb, uint32_t* subsdim, uint16_t* hash_table);

/** \brief Mappings for double particle replacing (different orbitals)
 *
 * Structure to directly map any configuration to others by replacing two
 * particles from two necessarily different orbitals. To build this
 * structure in a vector of integers, for each configuration compute
 * how many ways there are to remove two particles, and for each way
 * found there are `norb^2` orbitals to place them. Thus a vector of
 * `strides` is set, where `strides[i]` is the starting index in the
 * returned integer array where these jump mappings are recorded for
 * configuration with hash index `i`
 *
 * EXAMPLE : Given a configuration `i`, find the configuration `j` which
 * one particle less in both states 'k' and 's'(with s > k) and one more
 * in states 'q' and 'l'(replacement two particles)
 *
 * SOL : Using the mapping returned by this structure, we start in
 * `strides[i]`, then we loop ignoring all possible removal until
 * get to the orbital indexes `k` and `s`, jumping the inner chunks
 * related to particles replacement(`norb^2`). Consider `m` as the
 * pursued index, then
 *
 * \code
 * m = strides[i];
 * for h = 0 ... k - 1
 * {
 *     for g = h + 1 ... M - 1
 *     {
 *         if occupation on h and g are greater than 1 then
 *         {
 *             m = m + M * M;
 *         }
 *     }
 * }
 * for g = k + 1 ... s - 1
 * {
 *     if occupation on k and g are greater than 1 then
 *     {
 *         m = m + M * M;
 *     }
 * }
 * m = q + l * M;
 * j = map[m];
 * \endcode
 *
 * \note The last configuration hold all particles in the the orbital
 * with highest number, from which is not possible to remove particles
 * from two different orbitals. Due to this property, the last element
 * of `strides` is also the size of returned mapping array
 *
 * \param[in] npar       number of particles
 * \param[in] norb       number of orbitals
 * \param[in] subsdim    subspaces dimensions from `get_subspaces_dim`
 * \param[in] hash_table configurational space hashing table
 * \param[out] strides   Array with the same size of space dimension
 *                       See the detailed description above for more info
 * \return Array with mappings. See how to use in the description above
 */
uint32_t*
get_double_diffjump_map(
    uint16_t  npar,
    uint16_t  norb,
    uint32_t* subsdim,
    uint16_t* hash_table,
    uint32_t* strides);

/** \brief Mappings for two particle replacing (removed from same orbital)
 *
 * Equivalent to `get_double_diffjump_map` routine though with both particles
 * taken from the same orbital. Equivalently an array of strides is set which
 * mark the starting index of mappings for each configuration in `strides[i]`
 *
 * EXAMPLE : Given a configuration i, find configuration j which has
 * two particle less in state `k`, replaced in states `q` and `l`
 *
 * \code
 * m = strides[i];
 * for h = 0 ... k - 1
 * {
 *     if occupation on h are greater than 2 then
 *     {
 *         m = m + M * M;
 *     }
 * }
 * j = map[m + q + l*M];
 * \endcode
 *
 * \param[in] npar       number of particles
 * \param[in] norb       number of orbitals
 * \param[in] subsdim    subspaces dimensions from `get_subspaces_dim`
 * \param[in] hash_table configurational space hashing table
 * \param[out] strides   Array with the same size of space dimension
 *                       See the detailed description above for more info
 * \return Array with mappings. See how to use in the description above
 */
uint32_t*
get_double_equaljump_map(
    uint16_t  npar,
    uint16_t  norb,
    uint32_t* subsdim,
    uint16_t* hash_table,
    uint32_t* strides);

uint32_t
jump1_index(
    uint16_t  norb,
    uint32_t  dim,
    uint32_t  inp_index,
    uint32_t* map1jump,
    uint16_t  from_orb,
    uint16_t  to_orb);

uint32_t
jump2same_index(
    uint16_t  norb,
    uint16_t* conf,
    uint32_t  conf_stride,
    uint32_t* mapot,
    uint16_t  from_orb,
    uint16_t  to_orb1,
    uint16_t  to_orb2);

uint32_t
jump2diff_index(
    uint16_t  norb,
    uint16_t* conf,
    uint32_t  conf_stride,
    uint32_t* maptt,
    uint16_t  from_orb1,
    uint16_t  from_orb2,
    uint16_t  to_orb1,
    uint16_t  to_orb2);

#endif

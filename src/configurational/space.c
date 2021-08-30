#include "configurational/space.h"
#include "assistant/arrays_definition.h"
#include <stdio.h>
#include <stdlib.h>

/* ========================================================================
 *
 *    MODULE OF FUNCTIONS TO SETUP AND HANDLE CONFIGURATIONS(FOCK STATES)
 *
 *
 * A configuration is defined by a vector of integers, which refers  to the
 * occupation in each single particle state.  The  configurational space is
 * the spanned space by all configurations constrained to a fixed number of
 * particles N and single particle states M. The dimension of this space is
 * obtained by the combinatorial problem on how to fit  N balls in  M boxes
 * what yields the relation
 *
 *  dim_configurational  =  (N + M - 1)!
 *                          ------------
 *                          N!  (M - 1)!
 *
 * The functions presented here were made to generate, handle and operate
 * the configurational space.
 *
 *
 * ======================================================================== */

static void
overflow_dimension(uint16_t npar, uint16_t norb)
{
    printf(
        "\n\nINTEGER SIZE ERROR : overflow occurred "
        "representing the space dimension as 32-bit "
        "integers for %u particles and %u orbitals\n\n",
        npar,
        norb);
    exit(EXIT_FAILURE);
}

uint64_t
fac(uint8_t n)
{
    uint8_t  i;
    uint64_t nfac;
    nfac = 1;
    for (i = 1; i < n; i++)
    {
        if (nfac > UINT64_MAX / (i + 1))
        {
            printf(
                "\n\nINTEGER SIZE ERROR : overflow occurred computing "
                "factorial of %u as 64-bit integer\n\n",
                n);
            exit(EXIT_FAILURE);
        }
        nfac = nfac * (i + 1);
    }
    return nfac;
}

void
assert_space_parameters(uint16_t npar, uint16_t norb)
{
    if (npar == 0 || norb == 0)
    {
        printf(
            "\n\nInvalid number of particles %u or orbitals %u\n\n",
            npar,
            norb);
        exit(EXIT_FAILURE);
    }
    if (norb > MAX_ORBITALS)
    {
        printf(
            "\n\nERROR: Exceeded max number of orbitals %d\n\n", MAX_ORBITALS);
        exit(EXIT_FAILURE);
    }
}

uint32_t
space_dimension(uint16_t npar, uint16_t norb)
{
    uint32_t i, j, n;
    n = 1;
    j = 2;
    if (norb > npar)
    {
        for (i = npar + norb - 1; i > norb - 1; i--)
        {
            if (n > UINT32_MAX / i) overflow_dimension(npar, norb);
            n = n * i;
            if (n % j == 0 && j <= npar)
            {
                n = n / j;
                j = j + 1;
            }
        }
        for (i = j; i <= npar; i++) n = n / i;
        return ((int) n);
    }
    for (i = npar + norb - 1; i > npar; i--)
    {
        if (n > UINT32_MAX / i) overflow_dimension(npar, norb);
        n = n * i;
        if (n % j == 0 && j <= norb - 1)
        {
            n = n / j;
            j = j + 1;
        }
    }
    for (i = j; i < norb; i++) n = n / i;
    return n;
}

uint32_t*
get_subspaces_dim(uint16_t npar, uint16_t norb)
{
    uint16_t  i, j;
    uint32_t* subsdim;
    subsdim = get_uint32_array((norb + 1) * (npar + 1));
    for (i = 0; i < npar + 1; i++)
    {
        subsdim[i + (npar + 1) * 0] = 0;
        subsdim[i + (npar + 1) * 1] = 1;
        for (j = 2; j < norb + 1; j++)
        {
            subsdim[i + (npar + 1) * j] = space_dimension(i, j);
        }
    }
    return subsdim;
}

void
set_config(uint32_t hash_i, uint16_t npar, uint16_t norb, uint16_t* conf)
{
    uint16_t i, m;

    m = norb - 1;
    for (i = 0; i < norb; i++) conf[i] = 0;

    while (hash_i > 0)
    {
        // Check if can 'pay' the cost to set a particle
        // in current state. If not, try a 'cheaper' one
        while (hash_i < space_dimension(npar, m)) m = m - 1;
        hash_i = hash_i - space_dimension(npar, m);
        conf[m] = conf[m] + 1; // One more particle in orbital m
        npar = npar - 1;       // Less one particle to setup
    }
    // with hash_i zero set the rest in the first state
    if (npar > 0) conf[0] = conf[0] + npar;
}

uint32_t
hash_index(uint16_t npar, uint16_t norb, uint32_t* subsdim, uint16_t* conf)
{
    uint16_t m, n;
    uint32_t stride, hash_i;

    hash_i = 0;
    for (m = norb - 1; m > 0; m--)
    {
        n = conf[m];
        stride = (npar + 1) * m; // stride to access subspaces dimension
        while (n > 0)
        {
            hash_i = hash_i + subsdim[npar + stride * m];
            npar = npar - 1;
            n = n - 1;
        }
    }
    return hash_i;
}

uint16_t*
get_hash_table(uint16_t npar, uint16_t norb)
{
    uint32_t  hash_i, dim;
    uint16_t* hash_table;

    dim = space_dimension(npar, norb);
    if (dim > UINT32_MAX / norb)
    {
        printf("\n\nMEMORY ERROR : The size of the hashing "
               "table cannot be indexed by 32-bit integers\n\n");
        exit(EXIT_FAILURE);
    }
    hash_table = get_uint16_array(dim * norb);
    for (hash_i = 0; hash_i < dim; hash_i++)
    {
        set_config(hash_i, npar, norb, &hash_table[norb * hash_i]);
    }
    return hash_table;
}

uint32_t*
get_single_jump_map(
    uint16_t npar, uint16_t norb, uint32_t* subsdim, uint16_t* hash_table)
{
    uint16_t  q, k, l;
    uint32_t  hash_i, dim;
    uint16_t* conf;
    uint32_t* map;

    dim = space_dimension(npar, norb);
    if (dim > UINT32_MAX / norb / norb)
    {
        printf("\n\nMEMORY ERROR : The size of 1-body jump-mappings ");
        printf("cannot be indexed by 32-bit integers\n\n");
        exit(EXIT_FAILURE);
    }

    conf = get_uint16_array(norb);

    // The structure consider that for any configuration, there are
    // M^2 possible jumps among the individual particle states.  In
    // spite of the wasted elements from forbidden transitions that
    // are based on states that are empty, this is no problem  when
    // compared to routines that maps double jumps.
    map = get_uint32_array(norb * norb * dim);

    // Forbidden transitions will remain with 0 index when there
    // is a attempt to remove from a empty single particle state
    for (hash_i = 0; hash_i < dim * norb * norb; hash_i++) map[hash_i] = 0;

    for (hash_i = 0; hash_i < dim; hash_i++)
    {
        for (q = 0; q < norb; q++) conf[q] = hash_table[q + norb * hash_i];
        for (k = 0; k < norb; k++)
        {
            // check if there is at least a particle to remove
            if (conf[k] < 1) continue;
            for (l = 0; l < norb; l++)
            {
                conf[k] -= 1;
                conf[l] += 1;
                map[hash_i + k * dim + l * norb * dim] =
                    hash_index(npar, norb, subsdim, conf);
                conf[k] += 1;
                conf[l] -= 1;
            }
        }
    }
    free(conf);
    return map;
}

/** \brief Assistant function to allocate empty mapping */
static uint32_t*
alloc_double_diffjump_map(uint32_t dim, uint16_t norb, uint16_t* hash_table)
{
    /*********************************************************************
     * Structure allocation of mapping between two different Fock states *
     * whose the occupation numbers are related by jumps of  2 particles *
     * from different individual particle states                         *
     *                                                                   *
     * Given an non-empty orbital k, look for the next non-empty s > k   *
     * When found such a combination, it is necessary to allocate  M^2   *
     * new elements corresponding to the particles destiny, those that   *
     * were removed from states k and s                                  *
     ********************************************************************/
    uint32_t  i, possible_removals;
    uint16_t  k, s;
    uint32_t* map;

    possible_removals = 0;
    for (i = 0; i < dim; i++)
    {
        for (k = 0; k < norb; k++)
        {
            if (hash_table[k + i * norb] < 1) continue;
            for (s = k + 1; s < norb; s++)
            {
                if (hash_table[s + i * norb] < 1) continue;
                possible_removals++;
            }
        }
    }
    if (possible_removals > UINT32_MAX / norb / norb)
    {
        printf("\n\nMEMORY ERROR : Size of the 2-body jump-mappings ");
        printf("cannot be indexed by 32-bit integers\n\n");
        exit(EXIT_FAILURE);
    }
    map = get_uint32_array(possible_removals * norb * norb);
    for (i = 0; i < norb * norb * possible_removals; i++) map[i] = 0;
    return map;
}

uint32_t*
get_double_diffjump_map(
    uint16_t  npar,
    uint16_t  norb,
    uint32_t* subsdim,
    uint16_t* hash_table,
    uint32_t* strides)
{
    uint16_t  k, s, l, q;
    uint32_t  hash_i, dim, conf_rm, total_rm, inner_stride, map_index;
    uint16_t* conf;
    uint32_t* map;
    // total_rm is the total number of possible double removal of particles
    // conf_rm hold the number of double removals for current configuration
    // strides are computed by multiplying number of removals by norb^2

    dim = space_dimension(npar, norb);
    conf = get_uint16_array(norb);
    map = alloc_double_diffjump_map(dim, norb, hash_table);
    total_rm = 0;
    for (hash_i = 0; hash_i < dim; hash_i++)
    {
        for (k = 0; k < norb; k++) conf[k] = hash_table[k + norb * hash_i];
        // track the starting index for each configuration mapping
        strides[hash_i] = total_rm * norb * norb;
        conf_rm = 0;
        for (k = 0; k < norb; k++)
        {
            if (conf[k] < 1) continue;
            for (s = k + 1; s < norb; s++)
            {
                if (conf[s] < 1) continue;
                // If it is possible to remove particles from k and s,
                // we need to setup a chunk that corresponds to all
                // possible allocations for these removed particles
                inner_stride = conf_rm * norb * norb;
                for (l = 0; l < norb; l++)
                {
                    for (q = 0; q < norb; q++)
                    {
                        map_index =
                            strides[hash_i] + inner_stride + (l + q * norb);
                        conf[k] -= 1;
                        conf[s] -= 1;
                        conf[l] += 1;
                        conf[q] += 1;
                        map[map_index] = hash_index(npar, norb, subsdim, conf);
                        conf[k] += 1;
                        conf[s] += 1;
                        conf[l] -= 1;
                        conf[q] -= 1;
                    }
                }
                // Update number of possible removals
                conf_rm++;  // for current configuration
                total_rm++; // at all
            }
        }
    }
    free(conf);
    // The final size of this mapping is given by last strides element
    return map;
}

/** \brief Assistant function to allocate empty mapping */
uint32_t*
alloc_double_equaljump_map(uint32_t dim, uint16_t norb, uint16_t* hash_table)
{
    /****************************************************************
     * Allocation of array for replacement of two particles removed *
     * from the same orbital. Scan orbital occupations and for each *
     * one with at least two particles add stride of size `norb^2`  *
     ****************************************************************/
    uint16_t  k;
    uint32_t  hash_i, possible_removals;
    uint32_t* map;

    possible_removals = 0;

    for (hash_i = 0; hash_i < dim; hash_i++)
    {
        for (k = 0; k < norb; k++)
        {
            if (hash_table[k + hash_i * norb] < 2) continue;
            possible_removals++;
        }
    }
    if (possible_removals > UINT32_MAX / norb / norb)
    {
        printf("\n\nMEMORY ERROR : Size of 2-body(same orbital) ");
        printf("jump-mappings cannot be indexed by 32-bit integers\n\n");
        exit(EXIT_FAILURE);
    }
    map = get_uint32_array(possible_removals * norb * norb);
    for (hash_i = 0; hash_i < norb * norb * possible_removals; hash_i++)
    {
        map[hash_i] = 0;
    }
    return map;
}

uint32_t*
get_double_equaljump_map(
    uint16_t  npar,
    uint16_t  norb,
    uint32_t* subsdim,
    uint16_t* hash_table,
    uint32_t* strides)
{
    uint16_t  q, k, l;
    uint32_t  hash_i, dim, total_rm, conf_rm, inner_stride, map_index;
    uint16_t* conf;
    uint32_t* map;

    conf = get_uint16_array(norb);
    dim = space_dimension(npar, norb);
    map = alloc_double_equaljump_map(dim, norb, hash_table);
    total_rm = 0;
    for (hash_i = 0; hash_i < dim; hash_i++)
    {
        strides[hash_i] = total_rm * norb * norb;

        for (k = 0; k < norb; k++) conf[k] = hash_table[k + hash_i * norb];

        conf_rm = 0;
        for (k = 0; k < norb; k++)
        {
            if (conf[k] < 2) continue;
            inner_stride = conf_rm * norb * norb;
            for (l = 0; l < norb; l++)
            {
                for (q = 0; q < norb; q++)
                {
                    map_index = strides[hash_i] + inner_stride + (l + q * norb);
                    conf[k] -= 2;
                    conf[l] += 1;
                    conf[q] += 1;
                    map[map_index] = hash_index(npar, norb, subsdim, conf);
                    conf[k] += 2;
                    conf[l] -= 1;
                    conf[q] -= 1;
                }
            }
            conf_rm++;
            total_rm++;
        }
    }
    free(conf);
    // The size of this mapping array is last `strides` element plus the
    // number of possible double jumps from the last configuration. From
    // the last configuration we can only remove particles from the last
    // orbital, them the size of returned array is last `strides` element
    // plus `norb^2`
    return map;
}

uint32_t
jump1_index(
    uint16_t  norb,
    uint32_t  dim,
    uint32_t  inp_index,
    uint32_t* map1jump,
    uint16_t  from_orb,
    uint16_t  to_orb)
{
    uint32_t map_index = inp_index + from_orb * dim + to_orb * dim * norb;
    return map1jump[map_index];
}

uint32_t
jump2same_index(
    uint16_t  norb,
    uint16_t* conf,
    uint32_t  conf_stride,
    uint32_t* mapot,
    uint16_t  from_orb,
    uint16_t  to_orb1,
    uint16_t  to_orb2)
{
    uint16_t prev_orb, possible_rm;
    uint32_t inner_stride;
    possible_rm = 0;
    // count how many possible double removal before reach wanted orbital
    for (prev_orb = 0; prev_orb < from_orb; prev_orb++)
    {
        if (conf[prev_orb] > 1) possible_rm++;
    }
    // inner config mapping stride to ignore until get wanted orbital
    inner_stride = possible_rm * norb * norb;
    return mapot[conf_stride + inner_stride + to_orb1 + to_orb2 * norb];
}

uint32_t
jump2diff_index(
    uint16_t  norb,
    uint16_t* conf,
    uint32_t  conf_stride,
    uint32_t* maptt,
    uint16_t  from_orb1,
    uint16_t  from_orb2,
    uint16_t  to_orb1,
    uint16_t  to_orb2)
{
    uint16_t h, g, possible_rm;
    uint32_t inner_stride;
    // fix the ordering to work as the required
    if (from_orb2 < from_orb1)
    {
        h = from_orb1;
        from_orb1 = from_orb2;
        from_orb2 = h;
    }
    // count how many possible removal pairs before reach wanted orbitals
    possible_rm = 0;
    for (h = 0; h < from_orb1; h++)
    {
        for (g = h + 1; g < norb; g++)
        {
            if (conf[h] > 0 && conf[g] > 0) possible_rm++;
        }
    }
    for (g = from_orb1 + 1; g < from_orb2; g++)
    {
        if (conf[g] > 0) possible_rm++;
    }
    // inner config mapping stride to ignore until get wanted orbital
    inner_stride = possible_rm * norb * norb;
    return maptt[conf_stride + inner_stride + to_orb1 + to_orb2 * norb];
}

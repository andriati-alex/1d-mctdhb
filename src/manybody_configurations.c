#include "manybody_configurations.h"





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





long fac(int n)
{

/** Compute  n! avoiding overflow of integer size **/

    long
        i,
        nfac;

    nfac = 1;

    for (i = 1; i < n; i++)   
    {
        if (nfac > INT_MAX / (i + 1))
        {
            printf("\n\n\n\tINTEGER SIZE ERROR : overflow occurred");
            printf(" representing a factorial as ");
            printf("integers of 32 bits\n\n");
            exit(EXIT_FAILURE);
        }
        nfac = nfac * (i + 1);
    }

    return nfac;
}



int NC(int N, int M)
{

/** Total Number of Configurations(NC) of  N particles in  M states. Limit
    the application to work with integer size and the implementation avoid
    possible overflow for too large spaces                             **/

    long
        j,
        i,
        n;

    n = 1;
    j = 2;

    if  (M > N)
    {
        for (i = N + M - 1; i > M - 1; i --)
        {
            if (n > INT_MAX / i)
            {
                printf("\n\n\nINTEGER SIZE ERROR : overflow occurred");
                printf(" representing the number of configurations as ");
                printf("integers of 32 bits\n\n");
                exit(EXIT_FAILURE);
            }
            n = n * i;
            if (n % j == 0 && j <= N)
            {
                n = n / j;
                j = j + 1;
            }
        }

        for (i = j; i < N + 1; i++) n = n / i;

        return ((int) n);
    }

    for (i = N + M - 1; i > N; i --)
    {
        if (n > INT_MAX / i)
        {
            printf("\n\n\nINTEGER SIZE ERROR : overflow occurred");
            printf(" representing the number of configurations as ");
            printf("integers of 32 bits\n\n");
            exit(EXIT_FAILURE);
        }
        n = n * i;
        if (n % j == 0 && j <= M - 1)
        {
            n = n / j;
            j = j + 1;
        }
    }

    for (i = j; i < M; i++) n = n / i;

    return ((int) n);
}





Iarray setupNCmat(int N, int M)
{

/** Matrix of all possible outcomes form NC function  with
  * NCmat[i + N*j] = NC(i,j), where i <= N and j <= M, the
  * number of particles and states respectively.
  *
  * This is an auxiliar structure to avoid calls of  NC function
  * many times when converting Fock states to indexes        **/

    int
        i,
        j;

    Iarray
        NCmat;

    NCmat = iarrDef((M + 1) * (N + 1));

    for (i = 0; i < N + 1; i++)
    {
        NCmat[i + (N+1)*0] = 0;
        NCmat[i + (N+1)*1] = 1;
        for (j = 2; j < M + 1; j++) NCmat[i + (N+1)*j] = NC(i,j);
    }

    return NCmat;
}





void IndexToFock(int k, int N, int M, Iarray v)
{

/** Given an integer index 0 < k < NC(N,M) setup on v
  * the corresponding Fock vector with v[j] being the
  * occupation number on state j.
  *
  * This routine corresponds to an implementation of Algorithm 2
  * of the article **/

    int
        i,
        m;

    m = M - 1;

    for (i = 0; i < M; i++) v[i] = 0;

    while ( k > 0 )
    {
        // Check if can 'pay' the cost to put  the  particle
        // in current state. If not, try a 'cheaper' one
        while ( k - NC(N,m) < 0 ) m = m - 1;

        k = k - NC(N,m); // subtract cost
        v[m] = v[m] + 1; // One more particle in orbital m
        N = N - 1;       // Less one particle to setup
    }

    // with k zero put the rest in the first state
    if ( N > 0 ) v[0] = v[0] + N;
}





int FockToIndex(int N, int M, Iarray NCmat, Iarray v)
{

/** Convert an occupation vector v to a integer number from
  * 0 to the NC(N,M) - 1. It uses the  NCmat  structure  to
  * avoid calls of NC function, see 'setupNCmat' function
  *
  * This routines is an implementation of algorithm 1 of the article **/

    int
        i,
        k,
        n,
        col;

    k = 0;
    col = N + 1; // stride to access matrix stored in vector

    // Empty one by one orbital starting from the last one
    for (i = M - 1; i > 0; i--)
    {
        n = v[i]; // Number of particles in a given orbital

        while (n > 0)
        {
            k = k + NCmat[N + col*i]; // number of combinations needed
            N = N - 1; // decrease total # of particles
            n = n - 1; // decrease # of particles in current state
        }
    }

    return k;
}





Iarray setupFocks(int N, int M)
{

/** A hashing table for the Fock states ordered.  Stores for each index
  * of configurations the occupation numbers of the corresponding  Fock
  * state, thus, replacing the usage of IndexToFock routine by a memory
  * access.
  *
  * ItoFock[j + k*M]  gives the occupation number of configuration k in
  * the orbital j **/

    int
        k,
        nc;

    Iarray
        ItoFock;

    nc = NC(N,M);

    if (nc > INT_MAX / M)
    {
        printf("\n\n\nMEMORY ERROR : Because of the size of the");
        printf(" hashing table it can't be indexed by 32-bit integers\n\n");
        exit(EXIT_FAILURE);
    }

    ItoFock = iarrDef(nc * M);

    for (k = 0; k < nc; k++)
    {
        IndexToFock(k,N,M,&ItoFock[M*k]);
    }

    return ItoFock;
}





Iarray OneOneMap(int N, int M, Iarray NCmat, Iarray IF)
{

/** Given the first configuration index, map it in  a second  one  which
  * the occupation vector differs from the first by a jump of a particle
  * from one state to another.
  *
  * Thus given the first index 'i' of a Fock state :
  *
  * Map[i + k * nc + l * nc * M] = index of another Fock state which
  * have one particle less in k that has been added in l
  *
  * It requires the NCmat and the Hashing table as arguments. See the
  * description of 'setupFocks' and 'setupNCmat' routines  above  **/

    int i,
        q,
        k,
        l,
        nc;

    Iarray
        v,
        Map;

    nc = NC(N,M);

    if (nc > INT_MAX / M / M)
    {
        printf("\n\n\nMEMORY ERROR : Because of the size of the");
        printf(" jump-mappings they can't be indexed by 32-bit integers\n\n");
        exit(EXIT_FAILURE);
    }
    
    v = iarrDef(M);

    // The structure consider that for any configuration, there are
    // M^2 possible jumps among the individual particle states.  In
    // spite of the wasted elements from forbidden transitions that
    // are based on states that are empty, this is no problem compared
    // to the routines that maps double jumps.
    Map = iarrDef(M * M * nc);

    // Forbidden transitions will remain with -1 value when there
    // is a attempt to remove from a empty  single particle state
    for (i = 0; i < nc * M * M; i++) Map[i] = -1;

    for (i = 0; i < nc; i++)
    {
        // Copy the occupation vector from i-th configuration
        for (q = 0; q < M; q++) v[q] = IF[q + M*i];

        for (k = 0; k < M; k++)
        {
            // check if there is at least a particle to remove
            if (v[k] < 1) continue;

            for (l = 0; l < M; l++)
            {
                // particle jump from state k to state l
                v[k] -= 1;
                v[l] += 1;
                Map[i + k * nc + l * M * nc] = FockToIndex(N,M,NCmat,v);
                v[k] += 1;
                v[l] -= 1;
            }
        }
    }

    free(v);

    return Map;
}





Iarray allocTwoTwoMap(int nc, int M, Iarray IF)
{

/** Structure allocation of mapping between two different Fock states
  * whose the occupation numbers are related by jumps of  2 particles
  * from different individual particle states
  *
  * Given an non-empty orbital k, look for the next non-empty s > k
  * When found such a combination, it is necessary to allocate  M^2
  * new elements corresponding to the particles destiny, those that
  * were removed from states k and s                            **/

    int
        i,
        k,
        s,
        chunks;

    Iarray
        Map;

    chunks = 0;

    for (i = 0; i < nc; i++)
    {

        for (k = 0; k < M; k++)
        {

            if (IF[k + i * M] < 1) continue;

            for (s = k + 1; s < M; s++)
            {

                if (IF[s + i * M] < 1) continue;

                chunks++;
            }
        }
    }

    if (chunks > INT_MAX / M / M)
    {
        printf("\n\n\nMEMORY ERROR : Because of the size of the");
        printf(" jump-mappings they can't be indexed by 32-bit integers\n\n");
        exit(EXIT_FAILURE);
    }

    Map = iarrDef(chunks * M * M);

    for (i = 0; i < M * M * chunks; i++) Map[i] = -1;

    return Map;
}





Iarray TwoTwoMap(int N, int M, Iarray NCmat, Iarray IF, Iarray strideC)
{

/** Structure to direct map a configuration to another by replacing
  * particle from two different orbitals. To build such a structure
  * in a vector of integers it looks in each configuration how many
  * different possibilities are to remove two particles  from  two
  * different states, and for  each time  it happens there are M^2
  * different places to put those particles. Thus this function also
  * has as output the last argument, vector strideC which  for  each
  * enumerated configuration i store the integer number, a index  of
  * the mapping where those possibilites to remove two particle starts.
  *
  * EXAMPLE : Given a configuration i, find configuration j which has
  * a particle less in states 'k' ans 's' (s > k),  and  two  more on
  * states 'q' and 'l'.
  *
  * SOL : Using the map returned by this structure we start  by  the
  * stride from the configuration i, excluding the mapped index from
  * all previous configurations. Then, walk in chunks of size M^2 until
  * reach the orbitals desired to remove the particles.
  *
  * m = strideC[i];
  *
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
  *
  * for g = k + 1 ... s - 1
  * {
  *     if occupation on k and g are greater than 1 then
  *     {
  *         m = m + M * M;
  *     }
  * }
  *
  * after adding the orbital strides proportional to the M^2 possibilities
  * where the particles removed can be placed, finish adding
  *
  * m = q + l * M;
  *
  * j = MapTT[m]; **/

    int
        i,
        k,
        s,
        l,
        q,
        nc,
        chunksO,
        chunksC,
        strideO,
        mapIndex;

    Iarray
        occ,
        Map;

    nc = NC(N,M);
    occ = iarrDef(M);

    Map = allocTwoTwoMap(nc,M,IF);

    chunksC = 0;

    for (i = 0; i < nc; i++)
    {
        // strideC[i] is the index where jumps based on Fock state  i begins
        // chunksC counts how many possibilities were found for the previous
        // Fock states, and for each possibility M^2 is the size of the chunk
        // because of the number of possible destinies for removed particles
        strideC[i] = chunksC * (M * M);

        for (k = 0; k < M; k++) occ[k] = IF[k + M * i];

        chunksO = 0;

        for (k = 0; k < M; k++)
        {

            if (occ[k] < 1) continue;

            for (s = k + 1; s < M; s++)
            {

                if (occ[s] < 1) continue;

                // If it is possible to remove particles from k and s
                // we need to setup a chunk that corresponds  to  all
                // possible destinies for the particles that are going
                // to be removed from k and s orbitals

                strideO = chunksO * M * M;

                for (l = 0; l < M; l++)
                {
                    // put one particle in l state
                    for (q = 0; q < M; q++)
                    {
                        // Put one particle in q state
                        mapIndex = strideC[i] + strideO + (l + q * M);
                        occ[k] -= 1;
                        occ[s] -= 1;
                        occ[l] += 1;
                        occ[q] += 1;
                        Map[mapIndex] = FockToIndex(N,M,NCmat,occ);
                        occ[k] += 1;
                        occ[s] += 1;
                        occ[l] -= 1;
                        occ[q] -= 1;
                    }
                }

                // New chunk set up. Update how many chunks have been done
                chunksO++; // for current configuration
                chunksC++; // at all
            }
        }
    }

    free(occ);

    // the final size of this mapping is given by the strideC[nc-1], since
    // the last configuration cannot have a double jump of particles  from
    // two different states.

    return Map;
}





Iarray allocOneTwoMap(int nc, int M, Iarray IF)
{

/** Analogously to allocTwoTwoMap create a structure for mapping between
  * two different Fock states, though here the occupation numbers are
  * related by jumps of 2 particles from the same orbital
  *
  * Given an orbital k that has at least 2 particles of a  configuration
  * there are M^2 possible orbitals to put these particle removed from k
  * For each time it happens among all configurations add a chunck of M^2
  * elements to the total size of the mapping array **/

    int
        i,
        k,
        chunks;

    Iarray
        Map;

    chunks = 0;

    for (i = 0; i < nc; i++)
    {

        for (k = 0; k < M; k++)
        {

            if (IF[k + i * M] < 2) continue;

            chunks++;

        }
    }

    if (chunks > INT_MAX / M / M)
    {
        printf("\n\n\nMEMORY ERROR : Because of the size of the");
        printf(" jump-mappings they can't be indexed by 32-bit integers\n\n");
        exit(EXIT_FAILURE);
    }

    Map = iarrDef(chunks * M * M);

    for (i = 0; i < M * M * chunks; i++) Map[i] = -1;

    return Map;
}





Iarray OneTwoMap(int N, int M, Iarray NCmat, Iarray IF, Iarray strideC)
{

/** Configure the mapping array of jump of 2 particle from the same orbital.
  * In contrast to the TwoTwoMap, for each configuration, for each orbital
  * that has more than 2 particles, store the index of configurations with
  * all possible destinies for the 2 removed particles.
  *
  * EXAMPLE : Given a configuration i, find configuration j which has
  * two particle less in state 'k', and place them in states 'q' and 'l'
  *
  * m = strideC[i];
  *
  * for h = 0 ... k - 1
  * {
  *     if occupation on h are greater than 2 then
  *     {
  *         m = m + M * M;
  *     }
  * }
  *
  * j = MapTT[m + q + l*M]; **/

    int
        i,
        q,
        k,
        l,
        nc,
        chunksO,
        chunksC,
        strideO,
        mapIndex;

    Iarray
        occ,
        Map;

    occ = iarrDef(M);
    nc = NC(N,M);

    Map = allocOneTwoMap(nc,M,IF);

    chunksC = 0;

    for (i = 0; i < nc; i++)
    {

        // stride for where the transitions of configuration i starts
        strideC[i] = chunksC * M * M;

        // Copy the occupation vector from C[i] coeff.
        for (k = 0; k < M; k++) occ[k] = IF[k + i * M];

        chunksO = 0;

        for (k = 0; k < M; k++)
        {

            // Must be able to remove two particles
            if (occ[k] < 2) continue;

            // jump a stride of previous transitions in this same conf.
            strideO = chunksO * M * M;

            for (l = 0; l < M; l++)
            {
                // put one particle in l state
                for (q = 0; q < M; q++)
                {
                    mapIndex = strideC[i] + strideO + (l + q * M);
                    // Put one particle in q state
                    occ[k] -= 2;
                    occ[l] += 1;
                    occ[q] += 1;
                    Map[mapIndex] = FockToIndex(N,M,NCmat,occ);
                    occ[k] += 2;
                    occ[l] -= 1;
                    occ[q] -= 1;
                }
            }
            chunksO++;
            chunksC++;
        }
    }

    free(occ);

    // The size of this mapping array is given by strideC[nc-1]  plus the
    // number of possible double jumps from the last configurations. From
    // the last configuration we can only remove particles from the  last
    // orbital, them the total size is strideC[nc-1] + M^2

    return Map;
}















/* ========================================================================
 *
 *                ONE-BODY DENSITY MATRIX < a*_k   a_l >
 *            ----------------------------------------------
 *
 * Once defined a set of Single-Particle States a many body state can be
 * expanded  in the configurational basis whose the vector that span the
 * space are named Fock states.  The one body density matrix is known as
 * the expected value of  1 creation and  1 annihilation operators for a
 * given many-body state.  The many-body state is here fully featured by
 * its coefficients in the configurational basis.
 *
 * ======================================================================== */

void OBrho(int N, int M, Iarray Map, Iarray IF, Carray C, Cmatrix rho)
{

    int i,
        j,
        k,
        l,
        vk,
        vl,
        nc;

    double
        mod2;

    double complex
        RHO;

    nc = NC(N,M);

    for (k = 0; k < M; k++)
    {

        // Diagonal elements

        RHO = 0;

        for (i = 0; i < nc; i++)
        {
            mod2 = creal(C[i]) * creal(C[i]) + cimag(C[i]) * cimag(C[i]);
            RHO = RHO + mod2 * IF[k + i*M];
        }

        rho[k][k] = RHO;

        // Off-diagonal elements

        for (l = k + 1; l < M; l++)
        {

            RHO = 0;

            for (i = 0; i < nc; i++)
            {
                vk = IF[k + M*i];
                if (vk < 1) continue;
                vl = IF[l + M*i];

                j = Map[i + k * nc + l * M * nc];
                RHO += conj(C[i]) * C[j] * sqrt((double)(vl+1) * vk);
            }

            // exploit the fact it is hermitian
            rho[k][l] = RHO;
            rho[l][k] = conj(RHO);
        }
    }

}










/* ========================================================================
 *
 *                    <   a*_k   a*_s   a_q   a_l   >
 *                    -------------------------------
 *
 * Setup the 4-indexed quantity above using the coefficients of the
 * many-body state expanded in the configurational basis
 *
 * ======================================================================== */

void TBrho(int N, int M, Iarray Map, Iarray MapOT, Iarray MapTT,
     Iarray strideOT, Iarray strideTT, Iarray IF, Carray C, Carray rho)
{

    int i, // int indices to number coeficients
        j,
        k,
        s,
        q,
        l,
        h,
        g,
        nc,
        M2,
        M3,
        vk,
        vs,
        vl,
        vq,
        chunks,
        strideOrb;

    double
        mod2,   // |Cj| ^ 2
        sqrtOf; // Factors from the action of creation/annihilation

    double complex
        RHO;





    // Auxiliar to memory access
    M2 = M * M;
    M3 = M * M * M;

    nc = NC(N,M);



    // Rule 1: Creation on k k / Annihilation on k k
    // =====================================================================
    for (k = 0; k < M; k++)
    {

        RHO = 0;

#pragma omp parallel for private(i,mod2) reduction(+:RHO)
        for (i = 0; i < nc; i++)
        {
            if (IF[k + i*M] < 2) continue;
            mod2 = creal(C[i]) * creal(C[i]) + cimag(C[i]) * cimag(C[i]);
            RHO  = RHO + mod2 * IF[k + i*M] * (IF[k + i*M] - 1);
        }

        rho[k + M * k + M2 * k + M3 * k] = RHO;
    }



    // Rule 2: Creation on k s / Annihilation on k s
    // =====================================================================
    for (k = 0; k < M; k++)
    {
        for (s = k + 1; s < M; s++)
        {

            RHO = 0;

#pragma omp parallel for private(i,mod2) reduction(+:RHO)
            for (i = 0; i < nc; i++)
            {
                mod2 = creal(C[i]) * creal(C[i]) + cimag(C[i]) * cimag(C[i]);
                RHO += mod2 * IF[k + i*M] * IF[s + i*M];
            }

            // commutation of bosonic operators is used
            // to fill elements by exchange  of indexes
            rho[k + s * M + k * M2 + s * M3] = RHO;
            rho[s + k * M + k * M2 + s * M3] = RHO;
            rho[s + k * M + s * M2 + k * M3] = RHO;
            rho[k + s * M + s * M2 + k * M3] = RHO;
        }
    }



    // Rule 3: Creation on k k / Annihilation on q q
    // =====================================================================
    for (k = 0; k < M; k++)
    {
        for (q = k + 1; q < M; q++)
        {

            RHO = 0;

#pragma omp parallel for private(i,j,h,vk,vq,chunks,strideOrb,sqrtOf) \
            reduction(+:RHO)
            for (i = 0; i < nc; i++)
            {
                h = i * M; // auxiliar stride for IF
                if (IF[h + k] < 2) continue;
                vk = IF[h + k];
                vq = IF[h + q];

                chunks = 0;
                for (j = 0; j < k; j++)
                {
                    if (IF[h + j] > 1) chunks++;
                }
                strideOrb = chunks * M * M;

                j = MapOT[strideOT[i] + strideOrb + q + q * M];

                sqrtOf = sqrt((double)(vk-1)*vk*(vq+1)*(vq+2));
                RHO += conj(C[i]) * C[j] * sqrtOf;
            }

            // Use 2-index-'hermiticity'
            rho[k + k * M + q * M2 + q * M3] = RHO;
            rho[q + q * M + k * M2 + k * M3] = conj(RHO);
        }
    }



    // Rule 4: Creation on k k / Annihilation on k l
    // =====================================================================
    for (k = 0; k < M; k++)
    {
        for (l = k + 1; l < M; l++)
        {

            RHO = 0;

#pragma omp parallel for private(i,j,vk,vl,sqrtOf) reduction(+:RHO)
            for (i = 0; i < nc; i++)
            {
                vk = IF[i*M + k];
                vl = IF[i*M + l];
                if (vk < 2) continue;

                j = Map[i + k * nc + l * M * nc];
                sqrtOf = (vk - 1) * sqrt((double) vk * (vl + 1));
                RHO += conj(C[i]) * C[j] * sqrtOf;
            }

            rho[k + k * M + k * M2 + l * M3] = RHO;
            rho[k + k * M + l * M2 + k * M3] = RHO;
            rho[l + k * M + k * M2 + k * M3] = conj(RHO);
            rho[k + l * M + k * M2 + k * M3] = conj(RHO);
        }
    }



    // Rule 5: Creation on k s / Annihilation on s s
    // =====================================================================
    for (k = 0; k < M; k++)
    {
        for (s = k + 1; s < M; s++)
        {

            RHO = 0;

#pragma omp parallel for private(i,j,vk,vs,sqrtOf) reduction(+:RHO)
            for (i = 0; i < nc; i++)
            {
                vk = IF[i*M + k];
                vs = IF[i*M + s];
                if (vk < 1 || vs < 1) continue;

                j = Map[i + k * nc + s * M * nc];
                sqrtOf = vs * sqrt((double) vk * (vs + 1));
                RHO += conj(C[i]) * C[j] * sqrtOf;
            }

            rho[k + s * M + s * M2 + s * M3] = RHO;
            rho[s + k * M + s * M2 + s * M3] = RHO;
            rho[s + s * M + s * M2 + k * M3] = conj(RHO);
            rho[s + s * M + k * M2 + s * M3] = conj(RHO);
        }
    }



    // Rule 6.0: Creation on k k / Annihilation on q l (k < q < l)
    // =====================================================================
    for (k = 0; k < M; k++)
    {
        for (q = k + 1; q < M; q++)
        {
            for (l = q + 1; l < M; l++)
            {

                RHO = 0;

#pragma omp parallel for private(i,j,h,vk,vq,vl,chunks,strideOrb,sqrtOf) \
            reduction(+:RHO)
                for (i = 0; i < nc; i++)
                {
                    h = i * M; // auxiliat stride for IF
                    if (IF[h + k] < 2) continue;
                    vk = IF[h + k];
                    vq = IF[h + q];
                    vl = IF[h + l];

                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (IF[h + j] > 1) chunks++;
                    }
                    strideOrb = chunks * M * M;

                    j = MapOT[strideOT[i] + strideOrb + l + q * M];

                    sqrtOf = sqrt((double)vk*(vk-1)*(vq+1)*(vl+1));

                    RHO += conj(C[i]) * C[j] * sqrtOf;
                }

                rho[k + k * M + q * M2 + l * M3] = RHO;
                rho[k + k * M + l * M2 + q * M3] = RHO;
                rho[l + q * M + k * M2 + k * M3] = conj(RHO);
                rho[q + l * M + k * M2 + k * M3] = conj(RHO);
            }
        }
    }



    // Rule 6.1: Creation on k k / Annihilation on q l (q < k < l)
    // =====================================================================
    for (q = 0; q < M; q++)
    {
        for (k = q + 1; k < M; k++)
        {
            for (l = k + 1; l < M; l++)
            {

                RHO = 0;

#pragma omp parallel for private(i,j,h,vk,vq,vl,chunks,strideOrb,sqrtOf) \
            reduction(+:RHO)
                for (i = 0; i < nc; i++)
                {
                    h = i * M;
                    if (IF[h + k] < 2) continue;
                    vk = IF[h + k];
                    vq = IF[h + q];
                    vl = IF[h + l];

                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (IF[h + j] > 1) chunks++;
                    }
                    strideOrb = chunks * M * M;

                    j = MapOT[strideOT[i] + strideOrb + l + q * M];

                    sqrtOf = sqrt((double)vk*(vk-1)*(vq+1)*(vl+1));

                    RHO += conj(C[i]) * C[j] * sqrtOf;
                }

                rho[k + k * M + q * M2 + l * M3] = RHO;
                rho[k + k * M + l * M2 + q * M3] = RHO;
                rho[l + q * M + k * M2 + k * M3] = conj(RHO);
                rho[q + l * M + k * M2 + k * M3] = conj(RHO);
            }
        }
    }



    // Rule 6.2: Creation on k k / Annihilation on q l (q < l < k)
    // =====================================================================
    for (q = 0; q < M; q++)
    {
        for (l = q + 1; l < M; l++)
        {
            for (k = l + 1; k < M; k++)
            {

                RHO = 0;

#pragma omp parallel for private(i,j,h,vk,vq,vl,chunks,strideOrb,sqrtOf) \
            reduction(+:RHO)
                for (i = 0; i < nc; i++)
                {
                    h = i * M; // auxiliar stride for IF
                    if (IF[h + k] < 2) continue;
                    vk = IF[h + k];
                    vq = IF[h + q];
                    vl = IF[h + l];

                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (IF[h + j] > 1) chunks++;
                    }
                    strideOrb = chunks * M * M;

                    j = MapOT[strideOT[i] + strideOrb + l + q * M];

                    sqrtOf = sqrt((double)vk*(vk-1)*(vq+1)*(vl+1));

                    RHO += conj(C[i]) * C[j] * sqrtOf;
                }

                rho[k + k * M + q * M2 + l * M3] = RHO;
                rho[k + k * M + l * M2 + q * M3] = RHO;
                rho[l + q * M + k * M2 + k * M3] = conj(RHO);
                rho[q + l * M + k * M2 + k * M3] = conj(RHO);
            }
        }
    }



    // Rule 7.0: Creation on k s / Annihilation on s l (s < k < l)
    // =====================================================================
    for (s = 0; s < M; s++)
    {
        for (k = s + 1; k < M; k++)
        {
            for (l = k + 1; l < M; l++)
            {

                RHO = 0;

#pragma omp parallel for private(i,j,vk,vs,vl,sqrtOf) reduction(+:RHO)
                for (i = 0; i < nc; i++)
                {
                    vs = IF[i*M + s];
                    vk = IF[i*M + k];
                    vl = IF[i*M + l];
                    if (vk < 1 || vs < 1) continue;
                    j = Map[i + k * nc + l * M * nc];
                    sqrtOf = vs * sqrt((double) vk * (vl + 1));
                    RHO += conj(C[i]) * C[j] * sqrtOf;
                }

                rho[k + s * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + l * M2 + s * M3] = RHO;
                rho[k + s * M + l * M2 + s * M3] = RHO;

                rho[l + s * M + s * M2 + k * M3] = conj(RHO);
                rho[s + l * M + s * M2 + k * M3] = conj(RHO);
                rho[s + l * M + k * M2 + s * M3] = conj(RHO);
                rho[l + s * M + k * M2 + s * M3] = conj(RHO);
            }
        }
    }



    // Rule 7.1: Creation on k s / Annihilation on s l (k < s < l)
    // =====================================================================
    for (k = 0; k < M; k++)
    {
        for (s = k + 1; s < M; s++)
        {
            for (l = s + 1; l < M; l++)
            {

                RHO = 0;

#pragma omp parallel for private(i,j,vk,vs,vl,sqrtOf) reduction(+:RHO)
                for (i = 0; i < nc; i++)
                {
                    vs = IF[i*M + s];
                    vk = IF[i*M + k];
                    vl = IF[i*M + l];
                    if (vk < 1 || vs < 1) continue;
                    j = Map[i + k * nc + l * M * nc];
                    sqrtOf = vs * sqrt((double) vk * (vl + 1));
                    RHO += conj(C[i]) * C[j] * sqrtOf;
                }

                rho[k + s * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + l * M2 + s * M3] = RHO;
                rho[k + s * M + l * M2 + s * M3] = RHO;

                rho[l + s * M + s * M2 + k * M3] = conj(RHO);
                rho[s + l * M + s * M2 + k * M3] = conj(RHO);
                rho[s + l * M + k * M2 + s * M3] = conj(RHO);
                rho[l + s * M + k * M2 + s * M3] = conj(RHO);
            }
        }
    }



    // Rule 7.2: Creation on k s / Annihilation on s l (k < l < s)
    // =====================================================================
    for (k = 0; k < M; k++)
    {
        for (l = k + 1; l < M; l++)
        {
            for (s = l + 1; s < M; s++)
            {

                RHO = 0;

#pragma omp parallel for private(i,j,vk,vs,vl,sqrtOf) reduction(+:RHO)
                for (i = 0; i < nc; i++)
                {
                    vs = IF[i*M + s];
                    vk = IF[i*M + k];
                    vl = IF[i*M + l];
                    if (vk < 1 || vs < 1) continue;
                    j = Map[i + k * nc + l * M * nc];
                    sqrtOf = vs * sqrt((double) vk * (vl + 1));
                    RHO += conj(C[i]) * C[j] * sqrtOf;
                }

                rho[k + s * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + s * M2 + l * M3] = RHO;
                rho[s + k * M + l * M2 + s * M3] = RHO;
                rho[k + s * M + l * M2 + s * M3] = RHO;

                rho[l + s * M + s * M2 + k * M3] = conj(RHO);
                rho[s + l * M + s * M2 + k * M3] = conj(RHO);
                rho[s + l * M + k * M2 + s * M3] = conj(RHO);
                rho[l + s * M + k * M2 + s * M3] = conj(RHO);
            }
        }
    }



    // Rule 8: Creation on k s / Annihilation on q l
    // =====================================================================
    for (k = 0; k < M; k++)
    {
        for (s = k + 1; s < M; s++)
        {
            for (q = 0; q < M; q++)
            {
                if (q == s || q == k) continue;

                for (l = q + 1; l < M; l ++)
                {

                    if (l == k || l == s) continue;

                    RHO = 0;

#pragma omp parallel for private(i,j,h,g,vk,vq,vl,vs,chunks,strideOrb,sqrtOf) \
            reduction(+:RHO)
                    for (i = 0; i < nc; i++)
                    {
                        j = i * M; // auxiliar stride for IF
                        if (IF[j + k] < 1 || IF[j + s] < 1) continue;

                        vk = IF[j + k];
                        vl = IF[j + l];
                        vq = IF[j + q];
                        vs = IF[j + s];

                        chunks = 0;
                        for (h = 0; h < k; h++)
                        {
                            for (g = h + 1; g < M; g++)
                            {
                                if (IF[h+j] > 0 && IF[g+j] > 0) chunks++;
                            }
                        }

                        for (g = k + 1; g < s; g++)
                        {
                            if (IF[g+j] > 0) chunks++;
                        }

                        strideOrb = chunks * M * M;

                        sqrtOf = sqrt((double)vk*vs*(vq+1)*(vl+1));

                        j = MapTT[strideTT[i] + strideOrb + q + l*M];

                        RHO += conj(C[i]) * C[j] * sqrtOf;
                    }

                    rho[k + s * M + q * M2 + l * M3] = RHO;
                    rho[s + k * M + q * M2 + l * M3] = RHO;
                    rho[k + s * M + l * M2 + q * M3] = RHO;
                    rho[s + k * M + l * M2 + q * M3] = RHO;
                }   // Finish l loop
            }       // Finish q loop
        }           // Finish s loop
    }               // Finish k loop

    /*       ------------------- END OF ROUTINE -------------------       */
}










/* ========================================================================
 *
 *                    APPLY THE MANY BODY HAMILTONIAN
 *                    -------------------------------
 *
 * One the many-body state can be expressed in the configurational basis
 * through the  basis expansion coefficients,  the Hamiltonian becomes a
 * matrix(complicated sparse one). Instead of requiring the matrix apply
 * the  creation and annihilation operator rules on  Fock  states to act
 * with the Hamiltonian
 *
 * ======================================================================== */

void applyHconf (int N, int M, Iarray Map, Iarray MapOT, Iarray MapTT,
     Iarray strideOT, Iarray strideTT, Iarray IF, Carray C, Cmatrix Ho,
     Carray Hint, Carray out)
{
    // Apply the many-body hamiltonian in a state expressed in
    // number-occupation basis with coefficients defined by C.



    int // Index of coeficients
        i,
        j,
        nc,
        chunks;

    int // enumerate orbitals
        h,
        g,
        k,
        l,
        s,
        q;

    int // auxiliar variables
        strideOrb,
        M2 = M * M,
        M3 = M * M * M;

    Iarray
        v;

    double
        sqrtOf;

    double complex
        z,
        w;

    nc = NC(N,M);


#pragma omp parallel private(i,j,k,s,q,l,h,g,chunks,strideOrb,z,w,sqrtOf,v)
    {

    v = iarrDef(M);

#pragma omp for schedule(static)
    for (i = 0; i < nc; i++)
    {
        w = 0;
        z = 0;

        for (k = 0; k < M; k++) v[k] = IF[k + i*M];
    
        /* ================================================================ *
         *                                                                  *
         *                       One-body contribution                      *
         *                                                                  *
         * ================================================================ */

        for (k = 0; k < M; k++)
        {
            if (v[k] < 1) continue;

            w = w + Ho[k][k] * v[k] * C[i];

            for (l = 0; l < M; l++)
            {
                if (l == k) continue;
                sqrtOf = sqrt((double)v[k] * (v[l] + 1));
                j = Map[i + k * nc + l * M * nc];
                w = w + Ho[k][l] * sqrtOf * C[j];
            }
        }


        /* ================================================================ *
         *                                                                  *
         *                       Two-body contribution                      *
         *                                                                  *
         * ================================================================ */



        // Rule 1: Creation on k k / Annihilation on k k
        // ==================================================================
        for (k = 0; k < M; k++)
        {
            sqrtOf = v[k] * (v[k] - 1);
            z += Hint[k + M * k + M2 * k + M3 * k] * C[i] * sqrtOf;
        }



        // Rule 2: Creation on k s / Annihilation on k s
        // ==================================================================
        for (k = 0; k < M; k++)
        {
            if (v[k] < 1) continue;
            for (s = k + 1; s < M; s++)
            {
                sqrtOf = v[k] * v[s];
                z += 4 * Hint[k + s*M + k*M2 + s*M3] * sqrtOf * C[i];
                /* WHY FACTOR 4 USED IN THE LINE ABOVE
                z += Hint[s + k*M + k*M2 + s*M3] * sqrtOf * C[i];
                z += Hint[s + k*M + s*M2 + k*M3] * sqrtOf * C[i];
                z += Hint[k + s*M + s*M2 + k*M3] * sqrtOf * C[i];
                */
            }
        }



        // Rule 3: Creation on k k / Annihilation on q q
        // ==================================================================
        for (k = 0; k < M; k++)
        {
            if (v[k] < 2) continue;
            for (q = 0; q < M; q++)
            {
                if (q == k) continue;
                sqrtOf = sqrt((double)(v[k]-1) * v[k] * (v[q]+1) * (v[q]+2));

                chunks = 0;
                for (j = 0; j < k; j++)
                {
                    if (v[j] > 1) chunks++;
                }
                strideOrb = chunks * M * M;
                j = MapOT[strideOT[i] + strideOrb + q + q * M];

                z += Hint[k + k * M + q * M2 + q * M3] * C[j] * sqrtOf;
            }
        }



        // Rule 4: Creation on k k / Annihilation on k l
        // ==================================================================
        for (k = 0; k < M; k++)
        {
            if (v[k] < 2) continue;
            for (l = 0; l < M; l++)
            {
                if (l == k) continue;
                sqrtOf = (v[k] - 1) * sqrt((double)v[k] * (v[l] + 1));
                j = Map[i + k * nc + l * M * nc];
                z += 2 * Hint[k + k * M + k * M2 + l * M3] * C[j] * sqrtOf;
                /* WHY FACTOR 2 IN THE LINE ABOVE
                z += Hint[k + k * M + l * M2 + k * M3] * C[j] * sqrtOf;
                */
            }
        }



        // Rule 5: Creation on k s / Annihilation on s s
        // ==================================================================
        for (k = 0; k < M; k++)
        {
            if (v[k] < 1) continue;
            for (s = 0; s < M; s++)
            {
                if (s == k || v[s] < 1) continue;
                sqrtOf = v[s] * sqrt((double)v[k] * (v[s] + 1));
                j = Map[i + k * nc + s * M * nc];
                z += 2 * Hint[k + s * M + s * M2 + s * M3] * C[j] * sqrtOf;
                /* WHY FACTOR 2 IN THE LINE ABOVE
                z += Hint[s + k * M + s * M2 + s * M3] * C[j] * sqrtOf;
                */
            }
        }



        // Rule 6.0: Creation on k k / Annihilation on q l (k < q < l)
        // ==================================================================
        for (k = 0; k < M; k++)
        {
            if (v[k] < 2) continue;
            for (q = k + 1; q < M; q++)
            {
                for (l = q + 1; l < M; l++)
                {
                    sqrtOf = sqrt((double)v[k]*(v[k]-1)*(v[q]+1)*(v[l]+1));

                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (v[j] > 1) chunks++;
                    }
                    strideOrb = chunks * M * M;

                    j = MapOT[strideOT[i] + strideOrb + q + l * M];
                    z += 2 * Hint[k + k*M + q*M2 + l*M3] * C[j] * sqrtOf;
                    /* WHY FACTOR 2 IN THE LINE ABOVE
                    z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    */
                }
            }
        }



        // Rule 6.1: Creation on k k / Annihilation on q l (q < k < l)
        // ==================================================================
        for (q = 0; q < M; q++)
        {
            for (k = q + 1; k < M; k++)
            {
                if (v[k] < 2) continue;
                for (l = k + 1; l < M; l++)
                {
                    sqrtOf = sqrt((double)v[k]*(v[k]-1)*(v[q]+1)*(v[l]+1));

                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (v[j] > 1) chunks++;
                    }
                    strideOrb = chunks * M * M;

                    j = MapOT[strideOT[i] + strideOrb + q + l * M];
                    z += 2 * Hint[k + k*M + q*M2 + l*M3] * C[j] * sqrtOf;
                    /* WHY FACTOR 2 IN THE LINE ABOVE
                    z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    */
                }
            }
        }



        // Rule 6.2: Creation on k k / Annihilation on q l (q < l < k)
        // ==================================================================
        for (q = 0; q < M; q++)
        {
            for (l = q + 1; l < M; l++)
            {
                for (k = l + 1; k < M; k++)
                {

                    if (v[k] < 2) continue;

                    sqrtOf = sqrt((double)v[k]*(v[k]-1)*(v[q]+1)*(v[l]+1));

                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (v[j] > 1) chunks++;
                    }
                    strideOrb = chunks * M * M;

                    j = MapOT[strideOT[i] + strideOrb + q + l * M];
                    z += 2 * Hint[k + k*M + q*M2 + l*M3] * C[j] * sqrtOf;
                    /* WHY FACTOR 2 IN THE LINE ABOVE
                    z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    */
                }
            }
        }



        // Rule 7.0: Creation on k s / Annihilation on q q (q > k > s)
        // ==================================================================
        for (q = 0; q < M; q++)
        {
            for (k = q + 1; k < M; k++)
            {
                if (v[k] < 1) continue;
                for (s = k + 1; s < M; s++)
                {
                    if (v[s] < 1) continue;
                    sqrtOf = sqrt((double)v[k] * v[s] * (v[q]+1) * (v[q]+2));

                    chunks = 0;
                    for (h = 0; h < k; h++)
                    {
                        for (g = h + 1; g < M; g++)
                        {
                            if (v[h] > 0 && v[g] > 0) chunks++;
                        }
                    }

                    for (g = k + 1; g < s; g++)
                    {
                        if (v[g] > 0) chunks++;
                    }

                    strideOrb = chunks * M * M;

                    j = MapTT[strideTT[i] + strideOrb + q + q*M];
                    z += 2 * Hint[k + s*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    /* WHY FACTOR 2 IN THE LINE ABOVE
                    z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    */
                }
            }
        }



        // Rule 7.1: Creation on k s / Annihilation on q q (k > q > s)
        // ==================================================================
        for (k = 0; k < M; k++)
        {
            if (v[k] < 1) continue;
            for (q = k + 1; q < M; q++)
            {
                for (s = q + 1; s < M; s++)
                {
                    if (v[s] < 1) continue;
                    sqrtOf = sqrt((double)v[k] * v[s] * (v[q]+1) * (v[q]+2));

                    chunks = 0;
                    for (h = 0; h < k; h++)
                    {
                        for (g = h + 1; g < M; g++)
                        {
                            if (v[h] > 0 && v[g] > 0) chunks++;
                        }
                    }

                    for (g = k + 1; g < s; g++)
                    {
                        if (v[g] > 0) chunks++;
                    }

                    strideOrb = chunks * M * M;

                    j = MapTT[strideTT[i] + strideOrb + q + q*M];
                    z += 2 * Hint[k + s*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    /* WHY FACTOR 2 IN THE LINE ABOVE
                    z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    */
                }
            }
        }



        // Rule 7.2: Creation on k s / Annihilation on q q (k > s > q)
        // ==================================================================
        for (k = 0; k < M; k++)
        {
            if (v[k] < 1) continue;
            for (s = k + 1; s < M; s++)
            {
                if (v[s] < 1) continue;
                for (q = s + 1; q < M; q++)
                {
                    sqrtOf = sqrt((double)v[k] * v[s] * (v[q]+1) * (v[q]+2));

                    chunks = 0;
                    for (h = 0; h < k; h++)
                    {
                        for (g = h + 1; g < M; g++)
                        {
                            if (v[h] > 0 && v[g] > 0) chunks++;
                        }
                    }

                    for (g = k + 1; g < s; g++)
                    {
                        if (v[g] > 0) chunks++;
                    }

                    strideOrb = chunks * M * M;

                    j = MapTT[strideTT[i] + strideOrb + q + q*M];
                    z += 2 * Hint[k + s*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    /* WHY FACTOR 2 IN THE LINE ABOVE
                    z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    */
                }
            }
        }



        // Rule 8: Creation on k s / Annihilation on s l
        // ==================================================================
        for (s = 0; s < M; s++)
        {
            if (v[s] < 1) continue; // may improve performance
            for (k = 0; k < M; k++)
            {
                if (v[k] < 1 || k == s) continue;
                for (l = 0; l < M; l++)
                {
                    if (l == k || l == s) continue;
                    sqrtOf = v[s] * sqrt((double)v[k] * (v[l] + 1));

                    j = Map[i + k * nc + l * M * nc];
                    z += 4 * Hint[k + s*M + s*M2 + l*M3] * C[j] * sqrtOf;
                    /* WHY FACTOR 4 IN THE LINE ABOVE
                    z += Hint[s + k*M + s*M2 + l*M3] * C[j] * sqrtOf;
                    z += Hint[s + k*M + l*M2 + s*M3] * C[j] * sqrtOf;
                    z += Hint[k + s*M + l*M2 + s*M3] * C[j] * sqrtOf;
                    */
                }
            }
        }



        // Rule 9: Creation on k s / Annihilation on q l
        // ==================================================================
        for (k = 0; k < M; k++)
        {
            if (v[k] < 1) continue;
            for (s = k + 1; s < M; s++)
            {
                if (v[s] < 1) continue;
                for (q = 0; q < M; q++)
                {
                    if (q == s || q == k) continue;
                    for (l = q + 1; l < M; l ++)
                    {
                        if (l == k || l == s) continue;
                        sqrtOf = sqrt((double)v[k]*v[s]*(v[q]+1)*(v[l]+1));

                        chunks = 0;
                        for (h = 0; h < k; h++)
                        {
                            for (g = h + 1; g < M; g++)
                            {
                                if (v[h] > 0 && v[g] > 0) chunks++;
                            }
                        }

                        for (g = k + 1; g < s; g++)
                        {
                            if (v[g] > 0) chunks++;
                        }

                        strideOrb = chunks * M * M;

                        j = MapTT[strideTT[i] + strideOrb + q + l*M];
                        z += 4 * Hint[k + s*M + q*M2 + l*M3] * C[j] * sqrtOf;
                        // Factor 4 corresponds to s > k and l > q instead
                        // of s != k and l != q

                    }   // Finish l
                }       // Finish q
            }           // Finish s
        }               // Finish k

        out[i] = w + 0.5 * z;
    }

    free(v);

    } // end of parallel region

}

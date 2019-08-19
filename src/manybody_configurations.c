#include "manybody_configurations.h"





/* ========================================================================
 *
 *       AUXILIAR FUNCTIONS TO CONFIGURE NUMBER OCCUPATION STATES
 *       --------------------------------------------------------
 *
 * A configuration is defined as one of the possibles occupation number
 * states (Fock vectors).  This is  a  combinatorial problem  on how to
 * fill  "M"  different Single-Particle States (SPS)  with N  available
 * particles.  The routines below  implements a mapping  between  these
 * Fock states and integer numbers, to address coefficients of a  many-
 * body state in Occupation Number Basis (ONB).
 *
 *
 * ======================================================================== */





long fac(int n)
{
    long
        i;

    long
        nfac;

    nfac = 1;

    for (i = 1; i < n; i++) nfac = nfac * (i + 1);

    return nfac;
}



int NC(int N, int M)
{

/** Number of Configurations(NC) of N particles in M states **/

    long
        i,
        n;

    n = 1;

    if  (M > N)
    {
        for (i = N + M - 1; i > M - 1; i --) n = n * i;
        return (int) (n / fac(N));
    }

    for (i = N + M - 1; i > N; i --) n = n * i;
    return (int) (n / fac(M - 1));
}





Iarray setupNCmat(int N, int M)
{

/** Matrix of all possible outcomes form NC function with
  * NCmat[i + (N+1)*j] = NC(i,j).
**/

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

/* Function to map a index to a configuration represented by a Fock state.
 * Given the index 'k', it will populate some orbital 'm' if it is greater
 * than all possible combinations over 'm-1' left behind.  After  add  one
 * particle in the orbital 'm' it is subtracted by NC(N,m-1),  the  number
 * of particle to be configured is reduced by one, and it  check  again if
 * the remaining value of 'k' is greater than the number  of  combinations
 * needed to populate the orbital 'm', and in  negative  case  proceed  to
 * populate orbital 'm-1'.
 */

    int
        i,
        m;

    m = M - 1;

    for (i = 0; i < M; i++) v[i] = 0;

    while ( k > 0 )
    {
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

/* Function to inverse map, configuration to index. It empty orbital by
 * orbital adding the number of configurations needed to put each
 * particle in the orbitals.
 */

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

/** All possible occupation vectors in a vector. To get the occupations
  * respect to configuration index k we  use  ItoFock[j + k*M]  with  j
  * going from 0 to M - 1 (the orbital number).
**/

    int
        k,
        nc;

    Iarray
        ItoFock;

    nc = NC(N,M);
    ItoFock = iarrDef(nc * M);

    for (k = 0; k < nc; k++)
    {
        IndexToFock(k,N,M,&ItoFock[M*k]);
    }

    return ItoFock;
}





Iarray OneOneMap(int N, int M, Iarray NCmat, Iarray IF)
{

/** Given a configuration index, map it in a new one which the
  * occupation vector differs from the first by a  jump  of  a
  * particle from one orital to another. Thus given i we have
  *
  * Map[i + k * nc + l * nc * M] = index of a configuration which
  * have one particle less in k that has been added in l.
**/

    int i,
        q,
        k,
        l,
        nc;

    Iarray
        v,
        Map;

    nc = NC(N,M);
    
    v = iarrDef(M);

    Map = iarrDef(M * M * nc);

    for (i = 0; i < nc * M * M; i++) Map[i] = -1;

    for (i = 0; i < nc; i++)
    {
        // Copy the occupation vector from C[i] coeff.

        for (q = 0; q < M; q++) v[q] = IF[q + M*i];

        for (k = 0; k < M; k++)
        {
            // Take one particle from k state
            if (v[k] < 1) continue;

            for (l = 0; l < M; l++)
            {
                // Put one particle in l state
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

    Map = iarrDef(chunks * M * M);

    for (i = 0; i < M * M * chunks; i++) Map[i] = -1;

    return Map;
}



Iarray TwoTwoMap(int N, int M, Iarray NCmat, Iarray IF, Iarray strideC)
{

/** From one configuration find another by removing one particle in two
  * different states and adding two in other two arbitrary  states.  To
  * build such a structure in a vector of integers  it  looks  in  each
  * configuration how many different possibilities  are  to  remove two
  * particles from two different states, and for  each time  it happens
  * there are M^2 different places to put those  particles.  Thus  this
  * function also has as output the last argument, vector strideC which
  * for each  enumerated configuration  i  store the integer number,  a
  * index of the mapping where those possibilites to remove two particle
  * starts.
  *
  * EXAMPLE : Given a configuration i, find configuration j which has
  * a particle less in states 'k' ans 's' (s > k),  and  two  more on
  * states 'q' and 'l'.
  *
  * SOL : Using the map returned by this structure we start by the
  * stride from the configuration i, so,
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
  * j = MapTT[m];
  *
  * -------------------------------------------------------------------------
  *
**/

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
        strideC[i] = chunksC * (M * M);

        for (k = 0; k < M; k++) occ[k] = IF[k + M * i];

        chunksO = 0;

        for (k = 0; k < M; k++)
        {

            if (occ[k] < 1) continue;

            for (s = k + 1; s < M; s++)
            {

                if (occ[s] < 1) continue;

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

                chunksO++;
                chunksC++;
            }
        }
    }

    free(occ);

    return Map;
}



Iarray allocOneTwoMap(int nc, int M, Iarray IF)
{

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

    Map = iarrDef(chunks * M * M);

    for (i = 0; i < M * M * chunks; i++) Map[i] = -1;

    return Map;
}



Iarray OneTwoMap(int N, int M, Iarray NCmat, Iarray IF, Iarray strideC)
{

/** From one configuration find another by removing two particle from a
  * state and adding two in other two arbitrary states. The strategy to
  * store the index of these transition between the states are  similar
  * to the described in TwoTwoMap function, but a bit more simpler.
  *
  * EXAMPLE : Given a configuration i, find configuration j which has
  * a two particle less in state 'k' and 's' (s > k), and place  them
  * in states 'q' and 'l'
  *
  * m = strideC[i];
  *
  * for h = 0 ... k - 1
  * {
  *     if occupation on h and g are greater than 2 then
  *     {
  *         m = m + M * M;
  *     }
  * }
  *
  * j = MapTT[m + q + l*M];
  *
  * -------------------------------------------------------------------------
  *
  */

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

        strideC[i] = chunksC * M * M;

        // Copy the occupation vector from C[i] coeff.
        for (k = 0; k < M; k++) occ[k] = IF[k + i * M];

        chunksO = 0;

        for (k = 0; k < M; k++)
        {

            // Must be able to remove two particles
            if (occ[k] < 2) continue;

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

    return Map;
}















/* ========================================================================
 *
 *                           <   a*_k   a_l   >
 *                    -------------------------------
 *
 * Once defined a set of Single-Particle Wave Functions (SPWF) a many
 * body state  can be expanded  in a  Occupation Number Configuration
 * Basis (ONCB) whose vector are also named Fock states. The one body
 * density matrix is known as the expected value of 1 creation  and 1
 * annihilation operators for a given many-body state.  Use the basis
 * to express the state and then compute using its coefficients (Cj).
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
 * Once defined a set of Single-Particle Wave Functions (SPWF) a many
 * body state  can be expanded  in a  Occupation Number Configuration
 * Basis (ONCB) whose vector are also named Fock states. The two body
 * density matrix is known as the expected value of 2 creation  and 2
 * annihilation operators for a given many-body state.  Use the basis
 * to express the state and then compute using its coefficients (Cj).
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



    /* ---------------------------------------------
     * Rule 1: Creation on k k / Annihilation on k k
    ------------------------------------------------------------------- */
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



    /* ---------------------------------------------
     * Rule 2: Creation on k s / Annihilation on k s
    ------------------------------------------------------------------- */
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



    /* ---------------------------------------------
     * Rule 3: Creation on k k / Annihilation on q q
    ------------------------------------------------------------------- */
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



    /* ---------------------------------------------
     * Rule 4: Creation on k k / Annihilation on k l
    ------------------------------------------------------------------- */
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



    /* ---------------------------------------------
     * Rule 5: Creation on k s / Annihilation on s s
    ------------------------------------------------------------------- */
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



    /* -----------------------------------------------------------
     * Rule 6.0: Creation on k k / Annihilation on q l (k < q < l)
    ------------------------------------------------------------------- */
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



    /* -----------------------------------------------------------
     * Rule 6.1: Creation on k k / Annihilation on q l (q < k < l)
    ------------------------------------------------------------------- */
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



    /* -----------------------------------------------------------
     * Rule 6.2: Creation on k k / Annihilation on q l (q < l < k)
    ------------------------------------------------------------------- */
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



    /* -----------------------------------------------------------
     * Rule 7.0: Creation on k s / Annihilation on s l (s < k < l)
    ------------------------------------------------------------------- */
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



    /* -----------------------------------------------------------
     * Rule 7.1: Creation on k s / Annihilation on s l (k < s < l)
    ------------------------------------------------------------------- */
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



    /* -----------------------------------------------------------
     * Rule 7.2: Creation on k s / Annihilation on s l (k < l < s)
    ------------------------------------------------------------------- */
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



    /* ---------------------------------------------
     * Rule 8: Creation on k s / Annihilation on q l
    ------------------------------------------------------------------- */
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
 * Once defined a set of Single-Particle Wave Functions (SPWF) a many
 * body state  can be expanded  in  a  Occupation Number Basis  (ONB)
 * whose vector are also named Fock states.Then to apply an  operator
 * on a state we need  its  coefficients in this basis  (Cj)  and the 
 * matrix elements of the operator that is done below.
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
                //v[k] -= 1;
                //v[l] += 1;
                //j = FockToIndex(N, M, NCmat, v);
                w = w + Ho[k][l] * sqrtOf * C[j];
                //v[k] += 1;
                //v[l] -= 1;
            }
        }


        /* ================================================================ *
         *                                                                  *
         *                       Two-body contribution                      *
         *                                                                  *
         * ================================================================ */


        /* ---------------------------------------------
         * Rule 1: Creation on k k / Annihilation on k k
        ------------------------------------------------------------------- */
        for (k = 0; k < M; k++)
        {
            sqrtOf = v[k] * (v[k] - 1);
            z += Hint[k + M * k + M2 * k + M3 * k] * C[i] * sqrtOf;
        }
        /* ---------------------------------------------------------------- */


        /* ---------------------------------------------
         * Rule 2: Creation on k s / Annihilation on k s
        ------------------------------------------------------------------- */
        for (k = 0; k < M; k++)
        {
            if (v[k] < 1) continue;
            for (s = k + 1; s < M; s++)
            {
                sqrtOf = v[k] * v[s];
                z += 4 * Hint[k + s*M + k*M2 + s*M3] * sqrtOf * C[i];
                /*
                z += Hint[s + k*M + k*M2 + s*M3] * sqrtOf * C[i];
                z += Hint[s + k*M + s*M2 + k*M3] * sqrtOf * C[i];
                z += Hint[k + s*M + s*M2 + k*M3] * sqrtOf * C[i];
                */
            }
        }
        /* ---------------------------------------------------------------- */


        /* ---------------------------------------------
         * Rule 3: Creation on k k / Annihilation on q q
        ------------------------------------------------------------------- */
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

                //v[k] -= 2;
                //v[q] += 2;
                // j = FockToIndex(N, M, NCmat, v);
                z += Hint[k + k * M + q * M2 + q * M3] * C[j] * sqrtOf;
                //v[k] += 2;
                //v[q] -= 2;
            }
        }
        /* ---------------------------------------------------------------- */


        /* ---------------------------------------------
         * Rule 4: Creation on k k / Annihilation on k l
        ------------------------------------------------------------------- */
        for (k = 0; k < M; k++)
        {
            if (v[k] < 2) continue;
            for (l = 0; l < M; l++)
            {
                if (l == k) continue;
                sqrtOf = (v[k] - 1) * sqrt((double)v[k] * (v[l] + 1));
                j = Map[i + k * nc + l * M * nc];
                //v[k] -= 1;
                //v[l] += 1;
                //j = FockToIndex(N, M, NCmat, v);
                z += 2 * Hint[k + k * M + k * M2 + l * M3] * C[j] * sqrtOf;
                // z += Hint[k + k * M + l * M2 + k * M3] * C[j] * sqrtOf;
                //v[k] += 1;
                //v[l] -= 1;
            }
        }
        /* ---------------------------------------------------------------- */


        /* ---------------------------------------------
         * Rule 5: Creation on k s / Annihilation on s s
        ------------------------------------------------------------------- */
        for (k = 0; k < M; k++)
        {
            if (v[k] < 1) continue;
            for (s = 0; s < M; s++)
            {
                if (s == k || v[s] < 1) continue;
                sqrtOf = v[s] * sqrt((double)v[k] * (v[s] + 1));
                j = Map[i + k * nc + s * M * nc];
                //v[k] -= 1;
                //v[s] += 1;
                //j = FockToIndex(N, M, NCmat, v);
                z += 2 * Hint[k + s * M + s * M2 + s * M3] * C[j] * sqrtOf;
                // z += Hint[s + k * M + s * M2 + s * M3] * C[j] * sqrtOf;
                //v[k] += 1;
                //v[s] -= 1;
            }
        }
        /* ---------------------------------------------------------------- */


        /* -----------------------------------------------------------
         * Rule 6.0: Creation on k k / Annihilation on q l (k < q < l)
        ------------------------------------------------------------------- */
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

                    //v[k] -= 2;
                    //v[l] += 1;
                    //v[q] += 1;
                    //j = FockToIndex(N, M, NCmat, v);
                    z += 2 * Hint[k + k*M + q*M2 + l*M3] * C[j] * sqrtOf;
                    // z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    //v[k] += 2;
                    //v[l] -= 1;
                    //v[q] -= 1;
                }
            }
        }
        /* ---------------------------------------------------------------- */


        /* -----------------------------------------------------------
         * Rule 6.1: Creation on k k / Annihilation on q l (q < k < l)
        ------------------------------------------------------------------- */
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

                    //v[k] -= 2;
                    //v[l] += 1;
                    //v[q] += 1;
                    //j = FockToIndex(N, M, NCmat, v);
                    z += 2 * Hint[k + k*M + q*M2 + l*M3] * C[j] * sqrtOf;
                    // z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    //v[k] += 2;
                    //v[l] -= 1;
                    //v[q] -= 1;
                }
            }
        }
        /* ---------------------------------------------------------------- */


        /* -----------------------------------------------------------
         * Rule 6.2: Creation on k k / Annihilation on q l (q < l < k)
        ------------------------------------------------------------------- */
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

                    //v[k] -= 2;
                    //v[l] += 1;
                    //v[q] += 1;
                    //j = FockToIndex(N, M, NCmat, v);
                    z += 2 * Hint[k + k*M + q*M2 + l*M3] * C[j] * sqrtOf;
                    // z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    //v[k] += 2;
                    //v[l] -= 1;
                    //v[q] -= 1;
                }
            }
        }
        /* ---------------------------------------------------------------- */


        /* -----------------------------------------------------------
         * Rule 7.0: Creation on k s / Annihilation on q q (q > k > s)
        ------------------------------------------------------------------- */
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

                    //v[k] -= 1;
                    //v[s] -= 1;
                    //v[q] += 2;
                    //j = FockToIndex(N, M, NCmat, v);

                    z += 2 * Hint[k + s*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    // z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;

                    //v[k] += 1;
                    //v[s] += 1;
                    //v[q] -= 2;
                }
            }
        }
        /* ---------------------------------------------------------------- */


        /* -----------------------------------------------------------
         * Rule 7.1: Creation on k s / Annihilation on q q (k > q > s)
        ------------------------------------------------------------------- */
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

                    //v[k] -= 1;
                    //v[s] -= 1;
                    //v[q] += 2;
                    //j = FockToIndex(N, M, NCmat, v);

                    z += 2 * Hint[k + s*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    // z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;

                    //v[k] += 1;
                    //v[s] += 1;
                    //v[q] -= 2;
                }
            }
        }
        /* ---------------------------------------------------------------- */


        /* -----------------------------------------------------------
         * Rule 7.2: Creation on k s / Annihilation on q q (k > s > q)
        ------------------------------------------------------------------- */
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

                    //v[k] -= 1;
                    //v[s] -= 1;
                    //v[q] += 2;
                    //j = FockToIndex(N, M, NCmat, v);

                    z += 2 * Hint[k + s*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    // z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;

                    //v[k] += 1;
                    //v[s] += 1;
                    //v[q] -= 2;
                }
            }
        }
        /* ---------------------------------------------------------------- */


        /* ---------------------------------------------
         * Rule 8: Creation on k s / Annihilation on s l
        ------------------------------------------------------------------- */
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
                    //v[k] -= 1;
                    //v[l] += 1;
                    //j = FockToIndex(N, M, NCmat, v);
                    z += 4 * Hint[k + s*M + s*M2 + l*M3] * C[j] * sqrtOf;
                    /*
                    z += Hint[s + k*M + s*M2 + l*M3] * C[j] * sqrtOf;
                    z += Hint[s + k*M + l*M2 + s*M3] * C[j] * sqrtOf;
                    z += Hint[k + s*M + l*M2 + s*M3] * C[j] * sqrtOf;
                    */
                    //v[k] += 1;
                    //v[l] -= 1;
                }
            }
        }


        /* ---------------------------------------------
         * Rule 9: Creation on k s / Annihilation on q l
        ------------------------------------------------------------------- */
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

                        //v[k] -= 1;
                        //v[s] -= 1;
                        //v[q] += 1;
                        //v[l] += 1;
                        //j = FockToIndex(N, M, NCmat, v);

                        z += 4 * Hint[k + s*M + q*M2 + l*M3] * C[j] * sqrtOf;

                        //v[k] += 1;
                        //v[s] += 1;
                        //v[q] -= 1;
                        //v[l] -= 1;

                    }   // Finish l
                }       // Finish q
            }           // Finish s
        }               // Finish k

        out[i] = w + 0.5 * z;
    }

    free(v);

    } // end of parallel region

}

#include "../include/MCTDHB_configurations.h"





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





int fac(int n)
{   // return n !
    int n_fac = 1;
    for (int i = 1; i < n; i++) n_fac = n_fac * (i + 1);
    return n_fac;
}





int NC(int N, int M)
{   // Return # of possible configurations of N particles in M orbitals
    int i, n = 1;
    if  (M > N)
    {
        for (i = N + M - 1; i > M - 1; i --) n = n * i;
        return n / fac(N);
    }
    else
    {
        for (i = N + M - 1; i > N; i --) n = n * i;
        return n / fac(M - 1);
    }
}





int ** MountNCmat(int N, int M)
{   // Matrix of all possible configurations with
    // # particles < N and M < # of orbitals
    int i,
        j;

    int ** NCmat = (int ** ) malloc((N + 1) * sizeof(int * ));

    for (i = 0; i < N + 1; i++)
    {
        NCmat[i] = (int * ) malloc((M + 1) * sizeof(int));
    }

    for (i = 0; i < N + 1; i++)
    {
        NCmat[i][0] = 0;
        NCmat[i][1] = 1;
        for (j = 2; j < M + 1; j++) NCmat[i][j] = NC( i , j );
    }

    return NCmat;
}





int ** MountFocks(int N, int M, int ** NCmat)
{   // All possible occupation vectors organized in rows
    int k;

    int ** ItoFock = (int **) malloc(NC(N, M) * sizeof(int *));

    for (k = 0; k < NC(N, M); k++)
    {
        ItoFock[k] = (int * ) malloc(M * sizeof(int));
        IndexToFock(k, N, M, NCmat, ItoFock[k]);
    }

    return ItoFock;
}





void IndexToFock(int k, int N, int M, int ** NCmat, int * v)
{
    int x;

    int  i,
         m = M - 1;

    for (i = 0; i < M; i++) v[i] = 0;

    /* -------------------------------------------------------------
     * Put the particles in orbitals while has combinations to spend
    ----------------------------------------------------------------- */
    while ( k > 0 )
    {
        while ( k - NCmat[N][m] < 0 ) m = m - 1;
        x = k - NCmat[N][m];
        while ( x >= 0 )
        {
            v[m] = v[m] + 1; // One more particle in orbital m
            N = N - 1;       // Less one particle to setup
            k = x;
            x = x - NCmat[N][m];
        }
    }

    /* -------------------------------------------------------------
     * Put the particles in orbitals while has combinations to spend
    ----------------------------------------------------------------- */
    for (i = N; i > 0; i--) v[0] = v[0] + 1;
}





int FockToIndex(int N, int M, int ** NCmat, int * v)
{
    int i, n;
    int k = 0;

    /* ---------------------------------------------------
     * Empty one by one orbital starting from the last one
    ------------------------------------------------------ */
    for (i = M - 1; i > 0; i--)
    {
        n = v[i]; // Number of particles in the orbital
        while (n > 0)
        {
            k = k + NCmat[N][i]; // number of combinations needed
            N = N - 1;           // decrease the number of particles
            n = n - 1;
        }
    }

    return k;
}





void JumpMapping(int N, int M, int ** NCmat, int ** IF, int * Map)
{
    int i,
        q,
        k,
        l,
        nc = NCmat[N][M],
        * v = (int *) malloc(M * sizeof(int));

    for (i = 0; i < nc; i++)
    {   // Copy the occupation vector from C[i] coeff.
        for (q = 0; q < M; q++) v[q] = IF[i][q];
        for (k = 0; k < M; k++)
        {   // Take one particle from k state
            if (v[k] < 1) continue;
            for (l = 0; l < M; l++)
            {   // Put one particle in l state
                v[k] -= 1;
                v[l] += 1;
                Map[i + k * nc + l * M * nc] = FockToIndex(N, M, NCmat, v);
                v[k] += 1;
                v[l] -= 1;
            }
        }
    }

    free(v);
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





void OBrho(int N, int M, int ** NCmat, int ** IF, Carray C, Cmatrix rho)
{

    int i,
        j,
        k,
        l,
        q,
        nc,
        * v;

    double
        mod2;

    double complex
        RHO;

    nc = NCmat[N][M];

    for (k = 0; k < M; k++)
    {

        // Diagonal elements
        // ------------------------------------------------------------------

        RHO = 0;

        #pragma omp parallel shared(k, nc, C, IF) private(i, mod2) \
        reduction(+:RHO)
        {
            #pragma omp for schedule(static)
            for (i = 0; i < nc; i++)
            {
                mod2 = creal(C[i]) * creal(C[i]) + cimag(C[i]) * cimag(C[i]);
                RHO = RHO + mod2 * IF[i][k];
            }
        }

        rho[k][k] = RHO;
        // ------------------------------------------------------------------



        // Off-diagonal elements (different terms from operators)
        // ------------------------------------------------------------------
        for (l = k + 1; l < M; l++)
        {

            RHO = 0;

            #pragma omp parallel shared(l,k,M,N,nc,C,NCmat,IF) \
            private(i, j, q, v) reduction(+:RHO)
            {
                v = (int *) malloc(M * sizeof(int));

                #pragma omp for schedule(static)
                for (i = 0; i < nc; i++)
                {
                    if (IF[i][k] < 1) continue;
                    for (q = 0; q < M; q++) v[q] = IF[i][q];
                    v[k] -= 1;
                    v[l] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    v[k] += 1;
                    v[l] -= 1;
                    RHO += conj(C[i]) * C[j] * sqrt((double)(v[l]+1) * v[k]);
                }

                free(v); // Each thread release its vector
            }

            rho[k][l] = RHO;
            rho[l][k] = conj(RHO); // hermiticity
        }
        // ------------------------------------------------------------------
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





void TBrho(int N, int M, int ** NCmat, int ** IF, Carray C, Carray rho)
{

    int i, // int indices to number coeficients
        j,
        k,
        s,
        q,
        l,
        t,
        nc,
        M2,
        M3,
        * v;
    
    double
        mod2,   // |Cj| ^ 2
        sqrtOf; // Factors from the action of creation/annihilation

    double complex
        RHO;





    // Auxiliar to memory access
    M2 = M * M;
    M3 = M * M * M;

    nc = NCmat[N][M];



    /* ---------------------------------------------
     * Rule 1: Creation on k k / Annihilation on k k
    ------------------------------------------------------------------- */
    for (k = 0; k < M; k++)
    {

        RHO = 0;

        #pragma omp parallel for private(i,mod2) reduction(+:RHO)
        for (i = 0; i < nc; i++)
        {
            mod2 = creal(C[i]) * creal(C[i]) + cimag(C[i]) * cimag(C[i]);
            RHO  = RHO + mod2 * IF[i][k] * (IF[i][k] - 1);
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
                RHO += mod2 * IF[i][k] * IF[i][s];
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

            #pragma omp parallel private(i,j,t,sqrtOf,v) reduction(+:RHO)
            {
                v = (int *) malloc(M * sizeof(int));

                #pragma omp for
                for (i = 0; i < nc; i++)
                {
                    if (IF[i][k] < 2) continue;
                    for (t = 0; t < M; t++) v[t] = IF[i][t];
                    sqrtOf = sqrt((double)(v[k]-1)*v[k]*(v[q]+1)*(v[q]+2));
                    v[k] -= 2;
                    v[q] += 2;
                    j = FockToIndex(N, M, NCmat, v);
                    RHO += conj(C[i]) * C[j] * sqrtOf;
                }

                free(v);
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

            #pragma omp parallel private(i,j,t,sqrtOf,v) reduction(+:RHO)
            {
                v = (int *) malloc(M * sizeof(int));

                #pragma omp for
                for (i = 0; i < nc; i++)
                {
                    if (IF[i][k] < 2) continue;
                    for (t = 0; t < M; t++) v[t] = IF[i][t];
                    sqrtOf = (v[k] - 1) * sqrt((double)v[k] * (v[l] + 1));
                    v[k] -= 1;
                    v[l] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    RHO += conj(C[i]) * C[j] * sqrtOf;
                }

                free(v);
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

            #pragma omp parallel private(i,j,t,sqrtOf,v) reduction(+:RHO)
            {
                v = (int *) malloc(M * sizeof(int));

                #pragma omp for
                for (i = 0; i < nc; i++)
                {
                    if (IF[i][k] < 1 || IF[i][s] < 1) continue;
                    for (t = 0; t < M; t++) v[t] = IF[i][t];
                    sqrtOf = v[s] * sqrt((double)v[k]*(v[s]+1));
                    v[k] -= 1;
                    v[s] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    RHO += conj(C[i]) * C[j] * sqrtOf;
                }

                free(v);
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

                #pragma omp parallel private(i,j,t,sqrtOf,v) reduction(+:RHO)
                {
                    v = (int *) malloc(M * sizeof(int));

                    #pragma omp for
                    for (i = 0; i < nc; i++)
                    {
                        if (IF[i][k] < 2) continue;
                        for (t = 0; t < M; t++) v[t] = IF[i][t];
                        sqrtOf = sqrt((double)v[k]*(v[k]-1)*(v[q]+1)*(v[l]+1));
                        v[k] -= 2;
                        v[l] += 1;
                        v[q] += 1;
                        j = FockToIndex(N, M, NCmat, v);
                        RHO += conj(C[i]) * C[j] * sqrtOf;
                    }

                    free(v);
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

                #pragma omp parallel private(i,j,t,sqrtOf,v) reduction(+:RHO)
                {
                    v = (int *) malloc(M * sizeof(int));

                    #pragma omp for
                    for (i = 0; i < nc; i++)
                    {
                        if (IF[i][k] < 2) continue;
                        for (t = 0; t < M; t++) v[t] = IF[i][t];
                        sqrtOf = sqrt((double)v[k]*(v[k]-1)*(v[q]+1)*(v[l]+1));
                        v[k] -= 2;
                        v[l] += 1;
                        v[q] += 1;
                        j = FockToIndex(N, M, NCmat, v);
                        RHO += conj(C[i]) * C[j] * sqrtOf;
                    }

                    free(v);
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

                #pragma omp parallel private(i,j,t,sqrtOf,v) reduction(+:RHO)
                {
                    v = (int *) malloc(M * sizeof(int));

                    #pragma omp for
                    for (i = 0; i < nc; i++)
                    {
                        if (IF[i][k] < 2) continue;
                        for (t = 0; t < M; t++) v[t] = IF[i][t];
                        sqrtOf = sqrt((double)v[k]*(v[k]-1)*(v[q]+1)*(v[l]+1));
                        v[k] -= 2;
                        v[l] += 1;
                        v[q] += 1;
                        j = FockToIndex(N, M, NCmat, v);
                        RHO += conj(C[i]) * C[j] * sqrtOf;
                    }

                    free(v);
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

                #pragma omp parallel private(i,j,t,sqrtOf,v) reduction(+:RHO)
                {
                    v = (int *) malloc(M * sizeof(int));

                    #pragma omp for
                    for (i = 0; i < nc; i++)
                    {
                        if (IF[i][k] < 1 || IF[i][s] < 1) continue;
                        for (t = 0; t < M; t++) v[t] = IF[i][t];
                        sqrtOf = v[s] * sqrt((double)v[k] * (v[l] + 1));
                        v[k] -= 1;
                        v[l] += 1;
                        j = FockToIndex(N, M, NCmat, v);
                        RHO += conj(C[i]) * C[j] * sqrtOf;
                    }

                    free(v);
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

                #pragma omp parallel private(i,j,t,sqrtOf,v) reduction(+:RHO)
                {
                    v = (int *) malloc(M * sizeof(int));

                    #pragma omp for
                    for (i = 0; i < nc; i++)
                    {
                        if (IF[i][k] < 1) continue;
                        for (t = 0; t < M; t++) v[t] = IF[i][t];
                        sqrtOf = v[s] * sqrt((double)v[k] * (v[l] + 1));
                        v[k] -= 1;
                        v[l] += 1;
                        j = FockToIndex(N, M, NCmat, v);
                        RHO += conj(C[i]) * C[j] * sqrtOf;
                    }

                    free(v);
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

                #pragma omp parallel private(i,j,t,sqrtOf,v) reduction(+:RHO)
                {
                    v = (int *) malloc(M * sizeof(int));
                    #pragma omp for
                    for (i = 0; i < nc; i++)
                    {
                        if (IF[i][k] < 1) continue;
                        for (t = 0; t < M; t++) v[t] = IF[i][t];
                        sqrtOf = v[s] * sqrt((double)v[k]*(v[l]+1));
                        v[k] -= 1;
                        v[l] += 1;
                        j = FockToIndex(N, M, NCmat, v);
                        RHO += conj(C[i]) * C[j] * sqrtOf;
                    }
                    free(v);
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
        for (s = 0; s < M; s++)
        {
            if (s == k) continue;

            for (q = 0; q < M; q++)
            {
                if (q == s || q == k) continue;

                for (l = 0; l < M; l ++)
                {

                    RHO = 0;

                    if (l == k || l == s || l == q) continue;

                    #pragma omp parallel private(i,j,t,sqrtOf,v) \
                    reduction(+:RHO)
                    {
                        v = (int *) malloc(M * sizeof(int));
                        #pragma omp for
                        for (i = 0; i < nc; i++)
                        {
                            if (IF[i][k] < 1 || IF[i][s] < 1) continue;
                            for (t = 0; t < M; t++) v[t] = IF[i][t];
                            sqrtOf = sqrt((double)v[k]*v[s]*(v[q]+1)*(v[l]+1));
                            v[k] -= 1;
                            v[s] -= 1;
                            v[q] += 1;
                            v[l] += 1;
                            j = FockToIndex(N, M, NCmat, v);
                            RHO += conj(C[i]) * C[j] * sqrtOf;
                        }
                        free(v);
                    }

                    rho[k + s * M + q * M2 + l * M3] = RHO;
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



void applyHconf (int N, int M, int ** NCmat, int ** IF, Carray C, Cmatrix Ho,
     Carray Hint, Carray out)
{
    // Apply the many-body hamiltonian in a state expressed in
    // number-occupation basis with coefficients defined by C.



    int // Index of coeficients
        i,
        j,
        nc = NCmat[N][M];



    int // enumerate orbitals
        k,
        l,
        s,
        q;

    /* ==================================================================== *
     *                                                                      *
     *                         Auxiliary variables                          *
     *                                                                      *
     * ==================================================================== */

    int
        M2 = M * M,
        M3 = M * M * M;


    int * v;             // Occupation vector on each iteration

    double sqrtOf;       // Factor from creation/annihilation operator

    double complex
        z,
        w;





    #pragma omp parallel firstprivate(N, M, M2, M3) \
    private(i, j, k, l, s, q, z, w, sqrtOf, v)
    {

    v = (int * ) malloc(M * sizeof(int));

    #pragma omp for schedule(static)
    for (i = 0; i < nc; i++)
    {
        w = 0;
        z = 0;

        for (k = 0; k < M; k++) v[k] = IF[i][k];
    
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
                v[k] -= 1;
                v[l] += 1;
                j = FockToIndex(N, M, NCmat, v);
                w = w + Ho[k][l] * sqrtOf * C[j];
                v[k] += 1;
                v[l] -= 1;
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
                v[k] -= 2;
                v[q] += 2;
                j = FockToIndex(N, M, NCmat, v);
                z += Hint[k + k * M + q * M2 + q * M3] * C[j] * sqrtOf;
                v[k] += 2;
                v[q] -= 2;
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
                v[k] -= 1;
                v[l] += 1;
                j = FockToIndex(N, M, NCmat, v);
                z += 2 * Hint[k + k * M + k * M2 + l * M3] * C[j] * sqrtOf;
                // z += Hint[k + k * M + l * M2 + k * M3] * C[j] * sqrtOf;
                v[k] += 1;
                v[l] -= 1;
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
                if (s == k) continue;
                sqrtOf = v[s] * sqrt((double)v[k] * (v[s] + 1));
                v[k] -= 1;
                v[s] += 1;
                j = FockToIndex(N, M, NCmat, v);
                z += 2 * Hint[k + s * M + s * M2 + s * M3] * C[j] * sqrtOf;
                // z += Hint[s + k * M + s * M2 + s * M3] * C[j] * sqrtOf;
                v[k] += 1;
                v[s] -= 1;
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
                    v[k] -= 2;
                    v[l] += 1;
                    v[q] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    z += 2 * Hint[k + k*M + q*M2 + l*M3] * C[j] * sqrtOf;
                    // z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    v[k] += 2;
                    v[l] -= 1;
                    v[q] -= 1;
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
                    v[k] -= 2;
                    v[l] += 1;
                    v[q] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    z += 2 * Hint[k + k*M + q*M2 + l*M3] * C[j] * sqrtOf;
                    // z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    v[k] += 2;
                    v[l] -= 1;
                    v[q] -= 1;
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
                    v[k] -= 2;
                    v[l] += 1;
                    v[q] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    z += 2 * Hint[k + k*M + q*M2 + l*M3] * C[j] * sqrtOf;
                    // z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    v[k] += 2;
                    v[l] -= 1;
                    v[q] -= 1;
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
                    v[k] -= 1;
                    v[s] -= 1;
                    v[q] += 2;
                    j = FockToIndex(N, M, NCmat, v);
                    z += 2 * Hint[k + s*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    // z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    v[k] += 1;
                    v[s] += 1;
                    v[q] -= 2;
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
                    v[k] -= 1;
                    v[s] -= 1;
                    v[q] += 2;
                    j = FockToIndex(N, M, NCmat, v);
                    z += 2 * Hint[k + s*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    // z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    v[k] += 1;
                    v[s] += 1;
                    v[q] -= 2;
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
                    v[k] -= 1;
                    v[s] -= 1;
                    v[q] += 2;
                    j = FockToIndex(N, M, NCmat, v);
                    z += 2 * Hint[k + s*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    // z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    v[k] += 1;
                    v[s] += 1;
                    v[q] -= 2;
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
                    v[k] -= 1;
                    v[l] += 1;
                    j = FockToIndex(N, M, NCmat, v);
                    z += 4 * Hint[k + s*M + s*M2 + l*M3] * C[j] * sqrtOf;
                    /*
                    z += Hint[s + k*M + s*M2 + l*M3] * C[j] * sqrtOf;
                    z += Hint[s + k*M + l*M2 + s*M3] * C[j] * sqrtOf;
                    z += Hint[k + s*M + l*M2 + s*M3] * C[j] * sqrtOf;
                    */
                    v[k] += 1;
                    v[l] -= 1;
                }
            }
        }


        /* ---------------------------------------------
         * Rule 9: Creation on k s / Annihilation on q l
        ------------------------------------------------------------------- */
        for (k = 0; k < M; k++)
        {
            if (v[k] < 1) continue;
            for (s = 0; s < M; s++)
            {
                if (v[s] < 1 || s == k) continue;
                for (q = 0; q < M; q++)
                {
                    if (q == s || q == k) continue;
                    for (l = 0; l < M; l ++)
                    {
                        if (l == k || l == s || l == q) continue;
                        sqrtOf = sqrt((double)v[k]*v[s]*(v[q]+1)*(v[l]+1));
                        v[k] -= 1;
                        v[s] -= 1;
                        v[q] += 1;
                        v[l] += 1;
                        j = FockToIndex(N, M, NCmat, v);
                        z += Hint[k + s*M + q*M2 + l*M3] * C[j] * sqrtOf;
                        v[k] += 1;
                        v[s] += 1;
                        v[q] -= 1;
                        v[l] -= 1;
                    }   // Finish l
                }       // Finish q
            }           // Finish s
        }               // Finish k

        out[i] = w + 0.5 * z;
    }

    free(v);

    } // End of parallel region

}

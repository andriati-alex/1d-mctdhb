#include "array_memory.h"



/* ========================================================================
 
                               MEMORY ALLOCATION                            

   ======================================================================== */



Iarray iarrDef(int n)
{
    int * ptr;

    ptr = (int * ) malloc( n * sizeof(int) );

    if (ptr == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for integer.");
        printf(" Size requested : %ld\n\n", n * sizeof(int));
        exit(EXIT_FAILURE);
    }

    return ptr;
}





Rarray rarrDef(int n)
{
    double * ptr;

    ptr = (double * ) malloc( n * sizeof(double) );

    if (ptr == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for double\n\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}





Carray carrDef(int n)
{
    double complex * ptr;

    ptr = (double complex * ) malloc( n * sizeof(double complex) );

    if (ptr == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for complex\n\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}





CMKLarray CMKLdef(int n)
{
    MKL_Complex16 * ptr;

    ptr = (MKL_Complex16 *) malloc( n * sizeof(MKL_Complex16) );

    if (ptr == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for complex(mkl)\n\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}





Rmatrix rmatDef(int m, int n)
{

/** Real matrix of m rows and n columns **/

    int i;

    double ** ptr;

    ptr = (double ** ) malloc( m * sizeof(double *) );

    if (ptr == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for (double *)\n\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < m; i++) ptr[i] = rarrDef(n);

    return ptr;
}





Cmatrix cmatDef(int m, int n)
{

/** Complex matrix of m rows and n columns **/

    int i;

    double complex ** ptr;

    ptr = (double complex ** ) malloc( m * sizeof(double complex *) );

    if (ptr == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for (complex *)\n\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < m; i++) ptr[i] = carrDef(n);

    return ptr;
}





CCSmat ccsmatDef(int n, int max_nonzeros)
{

/** Return empty CCS representation of matrix of n rows **/

    CCSmat M = (struct CCS *) malloc(sizeof(struct CCS));

    if (M == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for CCS structure\n\n");
        exit(EXIT_FAILURE);
    }

    M->m = max_nonzeros;
    M->vec = carrDef(max_nonzeros * n);
    M->col = (int *) malloc( max_nonzeros * n * sizeof(int) );

    if (M->col == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for integers\n\n");
        exit(EXIT_FAILURE);
    }

    return M;
}





/* ========================================================================
 
                               MEMORY RELEASE

   ======================================================================== */





void rmatFree(int m, Rmatrix M)
{

/** Release a real matrix of m rows **/

    int i;

    for (i = 0; i < m; i++) free(M[i]);

    free(M);
}





void cmatFree(int m, Cmatrix M)
{

/** Release a real matrix of m rows **/

    int i;

    for (i = 0; i < m; i++) free(M[i]);

    free(M);
}





void CCSFree(CCSmat M)
{

/** Release Compressed-Column Storage matrix **/

    free(M->col);
    free(M->vec);
    free(M);
}

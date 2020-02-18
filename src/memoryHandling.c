#include "memoryHandling.h"



/*********************************************************************
 *****************                                   *****************  
 *****************     MEMORY ALLOCATION SECTION     *****************
 *****************                                   *****************  
 *********************************************************************/

Iarray iarrDef(int n)
{
    int * ptr;

    ptr = (int * ) malloc( n * sizeof(int) );

    if (ptr == NULL)
    {
        printf("\n\n\nMEMORY ERROR : malloc fail for integer.");
        printf(" Elements requested : %ld\n\n",n);
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
        printf("\n\n\nMEMORY ERROR : malloc fail for double.");
        printf(" Elements requested : %ld\n\n",n);
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
        printf("\n\n\nMEMORY ERROR : malloc fail for complex.");
        printf(" Elements requested : %ld\n\n",n);
        exit(EXIT_FAILURE);
    }

    return ptr;
}



CMKLarray cmklDef(int n)
{
    MKL_Complex16 * ptr;

    ptr = (MKL_Complex16 *) malloc( n * sizeof(MKL_Complex16) );

    if (ptr == NULL)
    {
        printf("\n\n\nMEMORY ERROR : malloc fail for complex(mkl).");
        printf(" Elements requested : %ld\n\n",n);
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
        printf("\n\n\nMEMORY ERROR : malloc fail for (double *).");
        printf(" Elements requested : %ld\n\n",m);
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
        printf("\n\n\nMEMORY ERROR : malloc fail for (complex *).");
        printf(" Elements requested : %ld\n\n",m);
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < m; i++) ptr[i] = carrDef(n);

    return ptr;
}



CCSmat ccsmatDef(int n, int max_nonzeros)
{

/** Return empty CCS representation of matrix of n rows **/

    CCSmat M = (struct _CCSmat *) malloc(sizeof(struct _CCSmat));

    if (M == NULL)
    {
        printf("\n\n\nMEMORY ERROR : malloc fail for CCSmat structure\n\n");
        exit(EXIT_FAILURE);
    }

    M->m = max_nonzeros;
    M->vec = carrDef(max_nonzeros*n);
    M->col = iarrDef(max_nonzeros*n);

    return M;
}










/**********************************************************************
 ******************                                  ******************  
 ******************      MEMORY RELEASE SECTION      ******************
 ******************                                  ******************
 **********************************************************************/



void rarrFree(Rarray x)
{
    if (x == NULL)
    {
        printf("\n\n\nMEMORY ERROR : Tried to free NULL pointer ");
        printf(" of double\n\n");
        exit(EXIT_FAILURE);
    }
    free(x);
}



void carrFree(Carray x)
{
    if (x == NULL)
    {
        printf("\n\n\nMEMORY ERROR : Tried to free NULL pointer ");
        printf(" of complex\n\n");
        exit(EXIT_FAILURE);
    }
    free(x);
}



void iarrFree(Iarray x)
{
    if (x == NULL)
    {
        printf("\n\n\nMEMORY ERROR : Tried to free NULL pointer ");
        printf(" of integers\n\n");
        exit(EXIT_FAILURE);
    }
    free(x);
}



void rmatFree(int m, Rmatrix M)
{
    for (int i = 0; i < m; i++) rarrFree(M[i]);
    free(M);
}



void cmatFree(int m, Cmatrix M)
{
    for (int i = 0; i < m; i++) carrFree(M[i]);
    free(M);
}



void CCSFree(CCSmat M)
{
    iarrFree(M->col);
    carrFree(M->vec);
    free(M);
}



void ReleaseManyBodyDataPkg(ManyBodyPkg S)
{
    cmatFree(S->Morb,S->Ho);
    cmatFree(S->Morb,S->rho1);
    cmatFree(S->Morb,S->Omat);
    carrFree(S->C);
    carrFree(S->rho2);
    carrFree(S->Hint);
    free(S);
}



void ReleaseEqDataPkg(EqDataPkg MC)
{
    iarrFree(MC->IF);
    iarrFree(MC->NCmat);
    iarrFree(MC->Map);
    iarrFree(MC->MapOT);
    iarrFree(MC->MapTT);
    iarrFree(MC->strideOT);
    iarrFree(MC->strideTT);
    rarrFree(MC->V);
    free(MC);
}

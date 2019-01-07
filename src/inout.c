#include "inout.h"



void sepline()
{

/** print in current screen a separation line **/

    printf("\n=======================================");
    printf("=======================================\n");
}





/**     =============================================================

                             PRINTING ON SCREEN

        =============================================================     **/





void cprint(double complex z)
{

/** printf complex number in coordinates with 2 decimal digits**/

    printf("(%9.2E,%9.2E )", creal(z), cimag(z));
}





void carr_print(int n, Carray v)
{

/** Print in screen array of complex numbers as column vector **/

    int i;

    if (n < 30)
    {
        // print all numbers for short arrays
        for (i = 0; i < n; i++) { printf("\n\t"); cprint(v[i]); }
    }

    else
    {
        // print first and last 10 elements
        for (i = 0; i < 10; i++)     { printf("\n\t"); cprint(v[i]); }
        for (i = 0; i < 5; i++)      { printf("\n\t           .");   }
        for (i = n - 10; i < n; i++) { printf("\n\t"); cprint(v[i]); }
    }

    printf("\n");
}





void rarr_print(int n, Rarray v)
{

/** Print in screen array of complex numbers as column vector **/

    int i;

    if (n < 30)
    {
        // print all numbers for short arrays
        for (i = 0; i < n; i++) printf("\n\t%9.2E", v[i]);
    }

    else
    {
        // print first and last 10 elements
        for (i = 0; i < 10; i++)     printf("\n\t%9.2E", v[i]);
        for (i = 0; i < 5; i++)      printf("\n\t     .");
        for (i = n - 10; i < n; i++) printf("\n\t%9.2E", v[i]);
    }

    printf("\n");
}





void cmat_print(int m, int n, Cmatrix M)
{

    int
        i,
        j;

    for (i = 0; i < m; i++)
    {

        printf("\n");

        for (j = 0; j < n; j++)
        {
            printf("  ");
            cprint(M[i][j]);
        }
    }

    printf("\n");
}





void rmat_print(int m, int n, Rmatrix M)
{

    int
        i,
        j;

    for (i = 0; i < m; i++)
    {

        printf("\n");

        for (j = 0; j < n; j++)
        {
            printf("  ");
            printf("%9.2E", M[i][j]);
        }
    }

    printf("\n");
}





/**     =============================================================

                              PRINTING ON FILES

        =============================================================     **/





void carr_txt(char fname [], int M, Carray v)
{

/** Record a array of complex elements in a text file in a
  * suitable format to import as numpy array with python.
**/

    int
        j;

    double
        real,
        imag;

    FILE
        * data_file = fopen(fname, "w");



    if (data_file == NULL)
    {
        printf("\n\n\n\tERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }

    for (j = 0; j < M - 1; j ++)
    {

        real = creal(v[j]);
        imag = cimag(v[j]);

        if (imag >= 0) fprintf(data_file, "(%.15E+%.15Ej)", real, imag);
        else           fprintf(data_file, "(%.15E%.15Ej)", real, imag);

        fprintf(data_file, "\n");
    }

    real = creal(v[M-1]);
    imag = cimag(v[M-1]);

    if (imag >= 0) fprintf(data_file, "(%.15E+%.15Ej)", real, imag);
    else           fprintf(data_file, "(%.15E%.15Ej)", real, imag);

    fclose(data_file);
}





void rarr_txt(char fname [], int M, Rarray v)
{

/** Record array of real numbers as column vector in text file **/

    FILE
        * data_file = fopen(fname, "w");

    if (data_file == NULL)
    {
        printf("\n\n\n\tERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < M - 1; j ++)
    {
        fprintf(data_file, "%.15E", v[j]);
        fprintf(data_file, "\n");
    }

    fprintf(data_file, "%.15E", v[M-1]);

    fclose(data_file);
}





void carr_inline(FILE * f, int M, Carray v)
{

/** Given a opened file write complex array in current line of buffer **/

    int
        j;

    double
        real,
        imag;

    if (f == NULL)
    {
        printf("\n\n\n\tERROR: NULL file in carr_inline routine");
        printf(" in module src/inout.c\n\n");
        exit(EXIT_FAILURE);
    }

    for (j = 0; j < M; j ++)
    {

        real = creal(v[j]);
        imag = cimag(v[j]);

        if (imag >= 0) fprintf(f, "(%.15E+%.15Ej) ", real, imag);
        else           fprintf(f, "(%.15E%.15Ej) ", real, imag);

    }

    fprintf(f, "\n");
}





void rarr_inline(FILE * f, int M, Rarray v)
{

/** Given a opened file write complex array in current line of buffer **/

    int
        j;

    if (f == NULL)
    {
        printf("\n\n\n\tERROR: NULL file in carr_inline routine");
        printf(" in module src/inout.c\n\n");
        exit(EXIT_FAILURE);
    }

    for (j = 0; j < M; j ++) fprintf(f, "%.15E ", v[j]);

    fprintf(f, "\n");
}





void cmat_txt (char fname [], int m, int n, Cmatrix A)
{

    int
        i,
        j;

    double
        real,
        imag;

    FILE
        * f = fopen(fname, "w");



    if (f == NULL)
    {   // impossible to open file with the given name
        printf("\n\n\n\tERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }



    for (i = 0; i < m - 1; i++)
    {

        for (j = 0; j < n; j++)
        {

            real = creal(A[i][j]);
            imag = cimag(A[i][j]);

            if (imag >= 0) fprintf(f, "(%.15E+%.15Ej) ", real, imag);
            else           fprintf(f, "(%.15E%.15Ej) ", real, imag);
        }

        fprintf(f, "\n");
    }

    for (j = 0; j < n; j++)
    {

        real = creal(A[m-1][j]);
        imag = cimag(A[m-1][j]);

        if (imag >= 0) fprintf(f, "(%.15E+%.15Ej) ", real, imag);
        else           fprintf(f, "(%.15E%.15Ej) ", real, imag);
    }

    fclose(f);
}





void cmat_txt_T (char fname [], int m, int n, Cmatrix A)
{

    int
        i,
        j;

    double
        real,
        imag;

    FILE
        * f = fopen(fname, "w");



    if (f == NULL)
    {   // impossible to open file with the given name
        printf("\n\n\n\tERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }



    for (j = 0; j < n - 1; j++)
    {

        for (i = 0; i < m; i++)
        {

            real = creal(A[i][j]);
            imag = cimag(A[i][j]);

            if (imag >= 0) fprintf(f, "(%.15E+%.15Ej) ", real, imag);
            else           fprintf(f, "(%.15E%.15Ej) ", real, imag);
        }

        fprintf(f, "\n");
    }

    for (i = 0; i < m; i++)
    {

        real = creal(A[i][n-1]);
        imag = cimag(A[i][n-1]);

        if (imag >= 0) fprintf(f, "(%.15E+%.15Ej) ", real, imag);
        else           fprintf(f, "(%.15E%.15Ej) ", real, imag);
    }

    fclose(f);
}

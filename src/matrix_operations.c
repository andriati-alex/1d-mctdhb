#include "matrix_operations.h"



/*          ***********************************************

                        SETUP VALUES IN MATRIX

            ***********************************************          */



void cmatFill(int m, int n, double complex z, Cmatrix M)
{

/** Fill all matrix with a single constant value **/

    unsigned int
        i,
        j;

    for (i = 0; i < m; i++) { for (j = 0; j < n; j++) M[i][j] = z; }
}





void cmatFillDK(int n, int k, Carray z, Cmatrix M)
{

/** Fill a diagonal starting from k-column if k >= 0
  * Fill a diagonal starting from |k|-row  if k <  0
  *
  * of a square matrix of dimension 'n' with an array
  * that must have size of (n - |k|) at least.    **/

    unsigned int i;

    if (k >= 0) { for (i = 0; i < n - k; i++) M[i][i+k] = z[i]; }
    else        { for (i = 0; i < n + k; i++) M[i-k][i] = z[i]; }
}





void cmatFillTri(int n, Carray upper, Carray mid, Carray lower, Cmatrix M)
{

/** Fill a tridiagonal matrix **/

    cmatFill(n, n, 0, M);
    cmatFillDK(n, -1, lower, M);
    cmatFillDK(n,  1, upper, M);
    cmatFillDK(n,  0, mid, M);
}





void setValueCCS(int n, int i, int j, int col, double complex z, CCSmat M)
{

/** Put 'z' value in column 'col' and row 'i' being the j-th nonzero element
  * of the squared matrix M of size 'n' in CCS format **/

    M->vec[i + n * j] = z;
    M->col[i + n * j] = col;
}





CCSmat tri2CCS(int n, Carray upper, Carray lower, Carray mid)
{

/** Configure a CCS matrix from a tridigonal one, being passed through
  * 3  vectors  correponding to diagonals.  Return the address  of the
  * structure allocated **/

    unsigned int j;

    CCSmat M;

    M = ccsmatDef(n,3);

    // first and last rows have just 2 nonzero elements
    // then must be configured separately
    M->vec[0]         = mid[0];
    M->vec[n]         = upper[0];
    M->vec[2 * n]     = 0;
    M->vec[3 * n - 1] = 0;

    for (j = 1; j < n; j++)                 { M->vec[j] = lower[j - 1];     }
    for (j = n + 1; j < 2 * n; j++)         { M->vec[j] = mid[j - n];       }
    for (j = 2 * n + 1; j < 3 * n - 1; j++) { M->vec[j] = upper[j - 2 * n]; }



    // first and last rows have just 2 nonzero elements
    // then must be configured separately
    M->col[0]         = 0;
    M->col[n]         = 1;
    M->col[2 * n]     = 0;
    M->col[3 * n - 1] = 0;

    for (j = 1; j < n; j++)                 { M->col[j] = j - 1; }
    for (j = n + 1; j < 2 * n; j++)         { M->col[j] = j - n; }
    for (j = 2 * n + 1; j < 3 * n - 1; j++) { M->col[j] = j + 1 - 2 * n; }

    return M;
}





CCSmat cyclic2CCS(int n, Carray upper, Carray lower, Carray mid)
{

/** Configure a CCS matrix from a Cyclic tridigonal one, a matrix  with  three
  * diagonals and terms in the first row last column and last row first column
  * being passed through 3 vectors correponding to diagonals. The last term of
  * upper diagonal is taken as the nonzero(cyclic) element  at  the  top right
  * corner and the last element of lower represent the low left corner.
  * Return the address of the structure allocated **/

    unsigned int j;
    
    CCSmat M;

    M = ccsmatDef(n,3);

    M->vec[0]         = mid[0];
    M->vec[n]         = upper[0];
    M->vec[2 * n]     = upper[n-1]; // Cyclic term
    M->vec[n - 1]     = lower[n-1]; // Cyclic term
    M->vec[2 * n - 1] = lower[n-2];
    M->vec[3 * n - 1] = mid[n-1];

    for (j = 1; j < n - 1; j++)             { M->vec[j] = lower[j - 1];     }
    for (j = n + 1; j < 2 * n - 1; j++)     { M->vec[j] = mid[j - n];       }
    for (j = 2 * n + 1; j < 3 * n - 1; j++) { M->vec[j] = upper[j - 2 * n]; }
    


    M->col[0]         = 0;
    M->col[n]         = 1;
    M->col[2 * n]     = n - 1; // Cyclic term
    M->col[n - 1]     = 0;     // Cyclic Term
    M->col[2 * n - 1] = n - 2;
    M->col[3 * n - 1] = n - 1;

    for (j = 1; j < n - 1; j++)             { M->col[j] = j - 1;         }
    for (j = n + 1; j < 2 * n - 1; j++)     { M->col[j] = j - n;         }
    for (j = 2 * n + 1; j < 3 * n - 1; j++) { M->col[j] = j + 1 - 2 * n; }

    return M;
}





void RowMajor(int m, int n, Cmatrix M, Carray v)
{

/** Copy data from matrix to vector using row major layout
  * The size of v is required to be at least m*n       **/

    int i,
        j;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++) v[i * m + j] = M[i][j];
    }
}





CCSmat CNmat(int M, double dx, doublec dt, double a2, doublec a1, double inter,
       Rarray V, int cyclic, Carray upper, Carray lower, Carray mid)
{
  
/** Auxiliar routine to setup matrix elements from Crank-Nicolson
  * discretization scheme applied to linear part of PDE.
  *
  * Returns a pointer to structure of CCS matrix form of RHS
  *
  * Output Parameters:
  *     upper
  *     lower
  *     mid
  *
  * The three diagonals of LHS of the system of equations to solve
  *
**/



    CCSmat Mat;



/** Setup matrix to multiply initial vector (RHS of the linear system)
    ------------------------------------------------------------------ */
    // fill main diagonal (use upper as auxiliar pointer)
    carrFill(M - 1, - a2 * dt / dx / dx + I, upper);
    rcarrUpdate(M - 1, upper, dt / 2, V, mid);

    // fill upper diagonal
    carrFill(M - 1, a2 * dt / dx / dx / 2 + a1 * dt / dx / 4, upper);
    if (cyclic) { upper[M-2] = a2 * dt / dx / dx / 2 - a1 * dt / dx / 4; }
    else        { upper[M-2] = 0;                                        }

    // fill lower diagonal
    carrFill(M - 1, a2 * dt / dx / dx / 2 - a1 * dt / dx / 4, lower);
    if (cyclic) { lower[M-2] = a2 * dt / dx / dx / 2 + a1 * dt / dx / 4; }
    else        { lower[M-2] = 0;                                        }

    // Store in CCS format
    Mat = cyclic2CCS(M - 1, upper, lower, mid);



/** Setup matrix to multiply initial vector (RHS of the linear system)
    ------------------------------------------------------------------ */
    // fill main diagonal (use upper as auxiliar pointer)
    carrFill(M - 1, a2 * dt / dx /dx + I, upper);
    rcarrUpdate(M - 1, upper, -dt / 2, V, mid);

    // fill upper diagonal
    carrFill(M - 1, - a2 * dt / dx / dx / 2 - a1 * dt / dx / 4, upper);
    if (cyclic) { upper[M-2] = - a2 * dt / dx / dx / 2 + a1 * dt / dx / 4; }
    else        { upper[M-2] = 0;                                          }

    // fill lower diagonal
    carrFill(M - 1, - a2 * dt / dx / dx / 2 + a1 * dt / dx / 4, lower);
    if (cyclic) { lower[M-2] = - a2 * dt / dx / dx / 2 - a1 * dt / dx / 4; }
    else        { lower[M-2] = 0;                                          }



    return Mat;
}













/*          **********************************************

            MATRIX-VECTOR AND MATRIX-MATRIX MULTIPLICATION

            **********************************************          */



void cmatvec(int m, int n, Cmatrix M, Carray v, Carray ans)
{

    unsigned int
        i,
        j;

    for (i = 0; i < m; i++)
    {
        ans[i] = M[i][0] * v[0];
        for (j = 1; j < n; j++) ans[i] = ans[i] + M[i][j] * v[j];
    }
}





void cmatmat(int m, int n, int l, Cmatrix M, Cmatrix A, Cmatrix ans)
{

    unsigned int
        i,
        j,
        k;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < l; j++)
        {
            ans[i][j] = M[i][0] * A[0][j];
            for (k = 1; k < n; k++) ans[i][j] = ans[i][j] + M[i][k] * A[k][j];
        }
    }
}





void CCSvec(int n, Carray vals, int * cols, int m, Carray vec, Carray ans)
{

    unsigned int
        i,
        l;

    double complex re;

    #pragma omp parallel for private(l, i, re)
    for (i = 0; i < n; i++)
    {
        re = vals[i] * vec[cols[i]];
        for (l = 1; l < m; l++) re = re + vals[i + l*n] * vec[cols[i + l*n]];
        ans[i] = re;
    }
}



/*          ***********************************************

                          Inversion of matrices

            ***********************************************          */



int HermitianInv(int M, Cmatrix A, Cmatrix A_inv)
{

/** Use Lapack routine to solve systems of equations with the
  * right-hand-side being identity matrix  to get the inverse **/

    int i, // counter
        j, // counter
        l; // lapack success parameter

    int
        * ipiv;

    CMKLarray
        ArrayForm, // To call zhesv routine use row major layout of Matrix
        Id;        // Identity matrix in row major layout



    ipiv = (int *) malloc(M * sizeof(int));

    ArrayForm = CMKLdef(M * M);

    Id = CMKLdef(M * M);



    for (i = 0; i < M; i++)
    {
        // Setup (L)ower triangular part as a Row-Major-Array to use lapack
        ArrayForm[i * M + i].real = creal(A[i][i]);
        ArrayForm[i * M + i].imag = 0;
        Id[i * M + i].real = 1;
        Id[i * M + i].imag = 0;

        for (j = 0; j < i; j++)
        {
            ArrayForm[i * M + j].real = creal(A[i][j]);
            ArrayForm[i * M + j].imag = cimag(A[i][j]);

            ArrayForm[j * M + i].real = 0; // symbolic values
            ArrayForm[j * M + i].imag = 0; // for upper triangular part

            Id[i * M + j].real = 0;
            Id[i * M + j].imag = 0;

            Id[j * M + i].real = 0;
            Id[j * M + i].imag = 0;
        }
    }

    l = LAPACKE_zhesv(LAPACK_ROW_MAJOR, 'L', M, M, ArrayForm, M, ipiv, Id, M);

    for (i = 0; i < M; i++)
    {
        // Transcript the result back to matrix form
        for (j = 0; j < M; j++)
        {
            A_inv[i][j] = Id[i * M + j].real + I * Id[i * M + j].imag;
        }
    }

    free(ipiv);
    free(Id);
    free(ArrayForm);

    return l;
}

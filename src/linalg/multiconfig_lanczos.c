#include "linalg/multiconfig_lanczos.h"
#include "linalg/basic_linalg.h"
#include "assistant/arrays_definition.h"
#include "configurational/hamiltonian.h"
#include <math.h>
#include <mkl_lapacke.h>
#include <stdio.h>
#include <stdlib.h>

int
lanczos(
    MultiConfiguration multiconf,
    Cmatrix            Ho,
    Carray             Hint,
    int                lm,
    Carray             diag,
    Carray             offdiag,
    Cmatrix            lvec)
{

    int    i, j, k, nc, Npar, Morb;
    double tol, maxCheck;
    Carray out, ortho;

    nc = multiconf->dim;
    Npar = multiconf->npar;
    Morb = multiconf->norb;
    out = get_dcomplex_array(nc);
    ortho = get_dcomplex_array(lm);

    // Variables to check for a source of breakdown or numerical instability
    maxCheck = 0;
    tol = 1E-14;

    // Compute the first diagonal element of the resulting tridiagonal
    // matrix outside the main loop because there is a different  rule
    // that came up from Modified Gram-Schmidt orthogonalization in
    // Krylov space
    apply_hamiltonian(multiconf, lvec[0], Ho, Hint, out);
    diag[0] = carrDot(nc, lvec[0], out);

    for (i = 0; i < lm - 1; i++)
    {

        for (j = 0; j < nc; j++) out[j] = out[j] - diag[i] * lvec[i][j];
        // in the line above 'out' holds a new Lanczos vector but not
        // normalized, just orthogonal to the previous ones in 'exact
        // arithmetic'.

        // Additional re-orthogonalization procedure.  The Lanczos vectors
        // are suppose to form an orthonormal set, and thus when organized
        // in a matrix it forms an unitary  transformation up to numerical
        // precision 'Q'. Thus in addition, we subtract the QQ† applied to
        // the unnormalized new Lanczos vector we get above. Note that here
        // lvec has Lanczos vector organized by rows.
        for (k = 0; k < i + 1; k++) ortho[k] = carrDot(nc, lvec[k], out);

        for (j = 0; j < nc; j++)
        {
            for (k = 0; k < i + 1; k++) out[j] -= lvec[k][j] * ortho[k];
        }

        offdiag[i] = carrMod(nc, out);

        // Check up to numerical precision if it is safe  to  continue
        // This is equivalent to find a null vector and thus the basis
        // of Lanczos vectors from the initial guess given is  said to
        // have an invariant subspace of the operator
        if (maxCheck < creal(offdiag[i])) maxCheck = creal(offdiag[i]);
        if (creal(offdiag[i]) / maxCheck < tol) return (i + 1);

        // Compute new Lanczos vector by normalizing the orthogonal vector
        carrScalarMultiply(nc, out, 1.0 / offdiag[i], lvec[i + 1]);

        // Perform half of the operation to obtain a new diagonal element
        // of the tridiagonal system. See lines 9 and 10 from the ref.[1]
        apply_hamiltonian(multiconf, lvec[i + 1], Ho, Hint, out);
        for (j = 0; j < nc; j++)
        {
            out[j] = out[j] - offdiag[i] * lvec[i][j];
        }
        diag[i + 1] = carrDot(nc, lvec[i + 1], out);
    }

    free(ortho);
    free(out);

    return lm;
}

double
lowest_state_lanczos(
    MultiConfiguration multiconf,
    WorkspaceLanczos   lanczos_work,
    Cmatrix            Ho,
    Carray             Hint,
    Carray             C)
{

    /** GROUND STATE BY APPROXIMATIVE DIAGONALIZATION WITH LANCZOS ITERATIONS
        =====================================================================
        Use the routine implemented above for lanczos tridiagonal reduction  and
        the LAPACK dstev to diagonalize the unerlying tridiagonal system and get
        approximately the low lying(ground state) eigenvalue and eigenvector

        INPUT PARAMETERS :
            Niter - Suggested number of lanczos iterations
            MC - Multiconfigurational data package
            Orb - Fixed orbitals whose the configurational space is built on
            C - input for lanczos (first lanczos vector)

        OUTPUT PARAMETERS :
            C - End up with low lying eigenvector(ground state)

        RETURN :
            Low lying eigenvalue/ground state energy **/

    int i, k, j, nc, predictedIter, Niter;

    double sentinel, *d, *e, *eigvec;

    Carray diag, offdiag;

    Cmatrix lvec;

    nc = multiconf->dim;
    Niter = lanczos_work->iter;
    lvec = lanczos_work->lanczos_vectors;
    diag = lanczos_work->decomp_diag;
    offdiag = lanczos_work->decomp_offd;
    eigvec = lanczos_work->lapack_eigvec;
    e = lanczos_work->lapack_offd;
    d = lanczos_work->lapack_diag;

    offdiag[Niter - 1] = 0;
    carrCopy(nc, C, lvec[0]);

    // Call Lanczos to setup tridiagonal matrix and lanczos vectors
    predictedIter = Niter;
    Niter = lanczos(multiconf, Ho, Hint, Niter, diag, offdiag, lvec);
    if (Niter < predictedIter)
    {
        printf(
            "\n\nWARNING : lanczos iterations exit "
            "before expected with %d iterations\n\n",
            Niter);
    }

    // Transfer data to use lapack routine
    for (k = 0; k < Niter; k++)
    {
        if (fabs(cimag(diag[k])) > 1E-10)
        {
            printf("\n\nWARNING : Nonzero imaginary part in Lanczos\n\n");
        }
        d[k] = creal(diag[k]);    // Supposed to be real
        e[k] = creal(offdiag[k]); // Supposed to be real
        for (j = 0; j < Niter; j++) eigvec[k * Niter + j] = 0;
    }

    k = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'V', Niter, d, e, eigvec, Niter);
    if (k != 0)
    {
        printf("\n\nERROR IN DIAGONALIZATION\n\n");
        printf("LAPACK dstev routine returned %d\n\n", k);
        exit(EXIT_FAILURE);
    }

    sentinel = 1E15;
    // Get Index of smallest eigenvalue
    for (k = 0; k < Niter; k++)
    {
        if (sentinel > d[k])
        {
            sentinel = d[k];
            j = k;
        }
    }

    // Update C with the coefficients of ground state
    for (i = 0; i < nc; i++)
    {
        C[i] = 0;
        for (k = 0; k < Niter; k++) C[i] += lvec[k][i] * eigvec[k * Niter + j];
    }

    return sentinel;
}

#include "linalg/multiconfig_lanczos.h"
#include "assistant/arrays_definition.h"
#include "assistant/types_definition.h"
#include "configurational/hamiltonian.h"
#include "linalg/basic_linalg.h"
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
    uint16_t           niter,
    MultiConfiguration multiconf,
    Cmatrix            Ho,
    Carray             Hint,
    Carray             C)
{
    int              i, k, j, dim, predicted_iter;
    double           sentinel, *d, *e, *eigvec;
    Carray           diag, offdiag;
    Cmatrix          lvec;
    WorkspaceLanczos lanczos_work;

    lanczos_work = get_lanczos_workspace(niter, multiconf->dim);

    dim = multiconf->dim;
    niter = lanczos_work->iter;
    lvec = lanczos_work->lanczos_vectors;
    diag = lanczos_work->decomp_diag;
    offdiag = lanczos_work->decomp_offd;
    eigvec = lanczos_work->lapack_eigvec;
    e = lanczos_work->lapack_offd;
    d = lanczos_work->lapack_diag;

    offdiag[niter - 1] = 0;
    carrCopy(dim, C, lvec[0]);

    // Call Lanczos to setup tridiagonal matrix and lanczos vectors
    predicted_iter = niter;
    niter = lanczos(multiconf, Ho, Hint, niter, diag, offdiag, lvec);
    if (niter < predicted_iter)
    {
        printf(
            "\n\nWARNING : lanczos iterations exit "
            "before expected with %d iterations\n\n",
            niter);
    }

    // Transfer data to use lapack routine
    for (k = 0; k < niter; k++)
    {
        if (fabs(cimag(diag[k])) > 1E-10)
        {
            printf("\n\nWARNING : Nonzero imaginary part in Lanczos\n\n");
        }
        d[k] = creal(diag[k]);    // Supposed to be real
        e[k] = creal(offdiag[k]); // Supposed to be real
        for (j = 0; j < niter; j++) eigvec[k * niter + j] = 0;
    }

    k = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'V', niter, d, e, eigvec, niter);
    if (k != 0)
    {
        printf("\n\nERROR IN DIAGONALIZATION\n\n");
        printf("LAPACK dstev routine returned %d\n\n", k);
        exit(EXIT_FAILURE);
    }

    sentinel = 1E15;
    // Get Index of smallest eigenvalue
    for (k = 0; k < niter; k++)
    {
        if (sentinel > d[k])
        {
            sentinel = d[k];
            j = k;
        }
    }

    // Update C with the coefficients of ground state
    for (i = 0; i < dim; i++)
    {
        C[i] = 0;
        for (k = 0; k < niter; k++) C[i] += lvec[k][i] * eigvec[k * niter + j];
    }

    destroy_lanczos_workspace(lanczos_work);

    return sentinel;
}

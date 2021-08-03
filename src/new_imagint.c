#include "new_imagint.h"
#include "odesys.h"

void
set_rowmajor_from_matrix(int nrows, int ncols, Cmatrix mat, Carray vec)
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
            vec[i * ncols + j] = mat[i][j];
    }
}

void
set_matrix_from_rowmajor(int nrows, int ncols, Carray vec, Cmatrix mat)
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
            mat[i][j] = vec[i * ncols + j];
    }
}

void
dvr_odesys_tder(ComplexODEInputParameters inp, Carray orb_tder)
{
    EqDataPkg mc_space;
    ManyBodyPkg mb_state;
    _UltimateStruct* ultimate;
    ultimate = (_UltimateStruct*) inp->extra_args;
    mc_space = ultimate->space_struct;
    mb_state = ultimate->state_struct;
    set_matrix_from_rowmajor(
        mc_space->Morb, mc_space->Mpos, inp->y, ultimate->orb_input);
    imagderEXPDVR(
        mc_space,
        ultimate->orb_input,
        ultimate->orb_output,
        ultimate->rhoinv,
        mb_state->rho2,
        ultimate->dvr);
    for (int i = 0; i < mc_space->Morb; i++)
    {
        ultimate->orb_output[i][mc_space->Mpos - 1] =
            ultimate->orb_output[i][0];
    }
    set_rowmajor_from_matrix(
        mc_space->Morb, mc_space->Mpos, ultimate->orb_output, orb_tder);
    for (int i = 0; i < mc_space->Morb * mc_space->Mpos; i++)
    {
        orb_tder[i] = -I * orb_tder[i];
    }
}

int
new_imagint(EqDataPkg MC, ManyBodyPkg S, double dT, int Nsteps, int coefInteg)
{

    /** INTEGRATE EQUATIONS IN IMAGINARY TIME
        =====================================
        Integrate the MCTDHB equations in imaginary time to find the ground
        state. The integration is done using split-step technique  for  the
        orbitals, where the nonlinear/potential part is  evolved  with  2nd
        order Runge-Kutta scheme and the  linear  part  with crank-nicolson
        finite differences scheme. The coefficients may be chosen in job.conf
        file between Lanczos and 4th order Runge-Kutta.

        The many-body data in 'S' is updated until find the ground state or
        reach the requested number of steps                             **/

    int i, j, k, l, p, q, N, nc, Npar, Mpos, Morb, kmom, isTrapped;
    double eigFactor, R2, dx, a2, g, L;
    doublec E, a1, dt, vir, sum, diag, prevE;
    Rarray V, occ;
    Carray D1DVR, D2DVR, DerDVR, rk_input, rk_output;
    _ComplexWorkspaceRK rk_ws;
    _UltimateStruct ult;

    // FROM 1-BODY POTENTIAL DECIDE THE TYPE OF BOUNDARIES
    isTrapped = 0;
    if (strcmp(MC->Vname, "harmonic") == 0)
        isTrapped = 1;
    if (strcmp(MC->Vname, "doublewell") == 0)
        isTrapped = 1;
    if (strcmp(MC->Vname, "harmonicgauss") == 0)
        isTrapped = 1;

    // UNPACK DOMAIN PARAMETERS TO LOCAL VARIABLES
    nc   = MC->nc;
    Npar = MC->Npar;
    Mpos = MC->Mpos;
    Morb = MC->Morb;
    dx   = MC->dx;
    dt   = -I * dT;

    // UNPACK EQUATION COEFFICIENTS AND ONE-BODY POTENTIIAL
    a2 = MC->a2;
    a1 = MC->a1;
    g  = MC->g;
    V  = MC->V;

    // NUMBER OF DVR POINTS IGNORING LAST GRID POINT CONSIDERED AS
    // THE SAME OF THE FIRST ONE - PERIODIC BOUNDARY
    N = Mpos - 1;

    // LENGTH OF GRID DOMAIN
    L = N * dx;

    // REQUEST MEMORY ALLOCATION
    l      = Morb - 1;
    occ    = rarrDef(Morb);
    D2DVR  = carrDef(N * N); // second derivative matrix in DVR basis
    D1DVR  = carrDef(N * N); // unitary transformation to DVR basis
    DerDVR = carrDef(N * N); // add two matrices of derivatives

    // SETUP FIRST ORDER DERIVATIVE MATRIX IN DVR BASIS
    for (i = 0; i < N; i++)
    {
        // USE COMPLEX CONJ. TO COMPUTE UPPER TRIANGULAR PART ONLY j > i
        for (j = i + 1; j < N; j++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                kmom = k - N / 2;
                // NOTE THIS MINUS SIGN - I HAD NO EXPLANATION FOR IT
                diag = -(2 * I * PI * kmom / L);
                sum  = sum + diag * cexp(2 * I * kmom * (j - i) * PI / N) / N;
            }
            D1DVR[i * N + j] = sum;
            D1DVR[j * N + i] = -conj(sum);
        }
        // COMPUTE SEPARATELY THE DIAGONAL
        sum = 0;
        for (k = 0; k < N; k++)
        {
            kmom = k - N / 2;
            diag = (2 * I * PI * kmom / L);
            sum  = sum + diag / N;
        }
        D1DVR[i * N + i] = sum;
    }

    // SETUP SECOND ORDER DERIVATIVE MATRIX IN DVR BASIS
    // 'mom'entum variables because of exponential basis
    for (i = 0; i < N; i++)
    {
        // USE COMPLEX CONJ. TO COMPUTE UPPER TRIANGULAR PART ONLY j > i
        for (j = i + 1; j < N; j++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                kmom = k - N / 2;
                diag = -(2 * PI * kmom / L) * (2 * PI * kmom / L);
                sum  = sum + diag * cexp(2 * I * kmom * (j - i) * PI / N) / N;
            }
            D2DVR[i * N + j] = sum;
            D2DVR[j * N + i] = conj(sum);
        }
        // COMPUTE SEPARATELY THE DIAGONAL
        sum = 0;
        for (k = 0; k < N; k++)
        {
            kmom = k - N / 2;
            diag = -(2 * PI * kmom / L) * (2 * PI * kmom / L);
            sum  = sum + diag / N;
        }
        D2DVR[i * N + i] = sum;
    }

    // SETUP MATRIX CORRESPONDING TO DERIVATIVES ON HAMILTONIAN
    // INCLUDING THE EQUATION COEFFICIENTS IN FRONT OF THEM
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            DerDVR[i * N + j] = a2 * D2DVR[i * N + j] + a1 * D1DVR[i * N + j];
        }
    }

    // SETUP ONE/TWO-BODY HAMILTONIAN MATRIX ELEMENTS ON ORBITAL BASIS
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // SETUP ONE/TWO-BODY DENSITY MATRIX
    OBrho(Npar, Morb, MC->Map, MC->IF, S->C, S->rho1);
    TBrho(
        Npar,
        Morb,
        MC->Map,
        MC->MapOT,
        MC->MapTT,
        MC->strideOT,
        MC->strideTT,
        MC->IF,
        S->C,
        S->rho2);

    E         = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    prevE     = E;
    vir       = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
    R2        = MeanQuadraticR(MC, S->Omat, S->rho1);
    eigFactor = eigQuality(MC, S->C, S->Ho, S->Hint, E);
    hermitianEigvalues(Morb, S->rho1, occ);

    // OBSERVABLES FOR QUALITY CONTROL OF THE RESULTS
    printf("\n\nProgress    E/particle     sqrt<R^2>");
    printf("     |Virial/E|     |H[C]-E*C|    n0");
    sepline();
    printf("%5.1lf%%     %11.7lf", 0.0, creal(E));
    printf("     %7.4lf       %9.6lf", R2, cabs(vir / E));
    printf("      %8.5lf     %3.0lf%%", eigFactor, 100 * occ[l] / Npar);

    rk_ws.system_size = Morb * Mpos;
    rk_input          = carrDef(rk_ws.system_size);
    rk_output         = carrDef(rk_ws.system_size);
    alloc_cplx_rungekutta_wsarrays(&rk_ws);

    ult.space_struct = MC;
    ult.state_struct = S;
    ult.dvr          = DerDVR;
    ult.rhoinv       = cmatDef(Morb, Morb);
    ult.orb_input    = cmatDef(Morb, Mpos);
    ult.orb_output   = cmatDef(Morb, Mpos);

    // VARIATIONAL ORBITAL EVOLUTION
    for (i = 0; i < Nsteps; i++)
    {
        p = HermitianInv(Morb, S->rho1, ult.rhoinv);
        if (p != 0)
        {
            printf("\n\n\nFailed on Lapack inversion routine!\n");
            printf("-----------------------------------\n\n");

            printf("Matrix given was :\n");
            cmat_print(Morb, Morb, S->rho1);

            if (p > 0)
                printf("\nSingular decomposition : %d\n\n", p);
            else
                printf("\nInvalid argument given : %d\n\n", p);

            exit(EXIT_FAILURE);
        }
        set_rowmajor_from_matrix(Morb, Mpos, S->Omat, rk_input);
        cplx_rungekutta5(
            dT, 0.0, &dvr_odesys_tder, &ult, &rk_ws, rk_input, rk_output);
        set_matrix_from_rowmajor(Morb, Mpos, rk_output, S->Omat);
        // FORWARD ONE TIME STEP ORBITALS
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        // LOSS OF NORM => UNDEFINED BEHAVIOR ON ORTHOGONALITY
        Ortonormalize(Morb, Mpos, dx, S->Omat);

        // PROPAGATE COEFFICIENTS
        if (coefInteg < 2)
            coef_RK4(MC, S, dt);
        else
            LanczosIntegrator(coefInteg, MC, S->Ho, S->Hint, dt, S->C);
        // RENORMALIZE COEFICIENTS
        renormalizeVector(nc, S->C, 1.0);

        // Update quantities that depends on orbitals and coefficients
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar, Morb, MC->Map, MC->IF, S->C, S->rho1);
        TBrho(
            Npar,
            Morb,
            MC->Map,
            MC->MapOT,
            MC->MapTT,
            MC->strideOT,
            MC->strideTT,
            MC->IF,
            S->C,
            S->rho2);

        // CHECK IF THE BOUNDARIES ARE STILL GOOD FOR TRAPPED SYSTEMS
        if ((i + 1) % (Nsteps / 5) == 0 && isTrapped)
        {
            // After some time evolved check if initial domain is suitable
            // for the current working orbitals, to avoid oversized domain,
            // a useless length where the functions are zero anyway
            ResizeDomain(MC, S);
            dx = MC->dx;
            Ortonormalize(Morb, Mpos, dx, S->Omat);
            SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
            SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

            // RE-CONFIGURE DVR MATRICES
            L = N * dx;

            for (p = 0; p < N; p++)
            {
                for (q = 0; q < N; q++)
                {
                    sum = 0;
                    for (k = 0; k < N; k++)
                    {
                        kmom = k - N / 2;
                        diag = -(2 * I * PI * kmom / L);
                        sum  = sum +
                              diag * cexp(2 * I * kmom * (q - p) * PI / N) / N;
                    }
                    D1DVR[p * N + q] = sum;
                }
            }

            for (p = 0; p < N; p++)
            {
                for (q = 0; q < N; q++)
                {
                    sum = 0;
                    for (k = 0; k < N; k++)
                    {
                        kmom = k - N / 2;
                        diag = -(2 * PI * kmom / L) * (2 * PI * kmom / L);
                        sum  = sum +
                              diag * cexp(2 * I * kmom * (q - p) * PI / N) / N;
                    }
                    D2DVR[p * N + q] = sum;
                }
            }

            for (p = 0; p < N; p++)
            {
                for (q = 0; q < N; q++)
                {
                    DerDVR[p * N + q] =
                        a2 * D2DVR[p * N + q] + a1 * D1DVR[p * N + q];
                }
            }
        }

        E   = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
        vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
        R2  = MeanQuadraticR(MC, S->Omat, S->rho1);
        if ((i + 1) % (Nsteps / 500) == 0)
        {
            eigFactor = eigQuality(MC, S->C, S->Ho, S->Hint, E);
            hermitianEigvalues(Morb, S->rho1, occ);
            // Print in screen them on screen
            printf("\n%5.1lf%%     %11.7lf", (100.0 * i) / Nsteps, creal(E));
            printf("     %7.4lf       %9.6lf", R2, cabs(vir / E));
            printf("      %8.5lf     %3.0lf%%", eigFactor, 100 * occ[l] / Npar);
        }

        // CHECK IF THE ENERGY HAS STABILIZED TO STOP PROCESS
        // At every 200 time-steps performed
        if ((i + 1) % 200 == 0)
        {
            // Check if the energy stop decreasing and break the process
            if (fabs(creal(E - prevE) / creal(prevE)) < 5E-10)
            {
                free(D1DVR);
                free(D2DVR);
                free(DerDVR);

                // Check if "it is(good enough) eigenstate". In negative
                // case perform a final diagonalization
                eigFactor = eigQuality(MC, S->C, S->Ho, S->Hint, E);
                if (eigFactor > 1E-3)
                {
                    if (300 * nc < 5E7)
                    {
                        if (2 * nc / 3 < 300)
                            k = 2 * nc / 3;
                        else
                            k = 300;
                    } else
                        k = 5E7 / nc;

                    E = LanczosGround(k, MC, S->Omat, S->C) / Npar;
                }

                OBrho(Npar, Morb, MC->Map, MC->IF, S->C, S->rho1);
                TBrho(
                    Npar,
                    Morb,
                    MC->Map,
                    MC->MapOT,
                    MC->MapTT,
                    MC->strideOT,
                    MC->strideTT,
                    MC->IF,
                    S->C,
                    S->rho2);

                E   = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
                vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
                R2  = MeanQuadraticR(MC, S->Omat, S->rho1);
                eigFactor = eigQuality(MC, S->C, S->Ho, S->Hint, E);
                hermitianEigvalues(Morb, S->rho1, occ);

                printf("\n -END-     %11.7lf", creal(E));
                printf("     %7.4lf       %9.6lf", R2, cabs(vir / E));
                printf(
                    "      %8.5lf     %3.0lf%%",
                    eigFactor,
                    100 * occ[l] / Npar);

                free(occ);
                sepline();

                printf("\nProcess ended before because ");
                printf("energy stop decreasing.\n\n");

                return i + 1;
            }

            prevE = E;
        }
    }

    // FINISHED TIME EVOLUTION

    if (300 * nc < 5E7)
    {
        if (2 * nc / 3 < 300)
            k = 2 * nc / 3;
        else
            k = 300;
    } else
        k = 5E7 / nc;

    E = LanczosGround(k, MC, S->Omat, S->C) / Npar;
    renormalizeVector(nc, S->C, 1.0);

    sepline();
    printf("\nFinal E/particle = %.7lf\n", creal(E));
    printf("Process did not stop automatically because energy were");
    printf(" varying above accepted tolerance\n\n");

    free(rk_input);
    free(rk_output);
    free(occ);
    free(D2DVR);
    free(D1DVR);
    free(DerDVR);
    free_cplx_rungekutta_wsarrays(&rk_ws);
    cmatFree(Morb, ult.rhoinv);
    cmatFree(Morb, ult.orb_input);
    cmatFree(Morb, ult.orb_output);

    return Nsteps + 1;
}

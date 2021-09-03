#include "imagtimeIntegrator.h"




void imagNLTRAP_dOdt(EqDataPkg MC, Cmatrix Orb, Cmatrix dOdt,
     Cmatrix Ho, Carray Hint, Cmatrix rho1, Carray rho2)
{

/** TIME DERIVATIVE ACCORDING TO NONLINEAR + LINEAR POTENTIAL
    =========================================================
    Compute essentially the nonlinear part of the equation together with
    the 1-body(trap) potential, multiplied by minus imaginary  unit.  In
    split-step techniques give us the derivative when solving  nonlinear
    part alone. Since it includes the 1-body(trap) potential it  is more
    useful when solving the derivatives part by  spectral  methods.  The
    1-body density matrix inversion is evaluated here

    OUTPUT ARGUMENT : dOdt **/

    int
        k,
        s,
        j,
        M  = MC->Morb,
        Mpos  = MC->Mpos;



    double
        dx = MC->dx,
        * V = MC->V;



    double complex
        Proj,
        g = MC->g;



    Cmatrix
        rho_inv = cmatDef(M,M);



    // Invert the matrix and check if the operation was successfull
    s = HermitianInv(M,rho1,rho_inv);
    if (s != 0)
    {
        printf("\n\n\n\nFailed on Lapack inversion routine!\n");
        printf("-----------------------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(M,M,rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n",s);
        else       printf("\nInvalid argument given : %d\n\n",s);

        exit(EXIT_FAILURE);
    }



    // Update k-th orbital at discretized position j
    #pragma omp parallel for private(k,j) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
            dOdt[k][j] = - I * ( V[j] * Orb[k][j] + \
            nonlinear(M,k,j,g,Orb,rho_inv,rho2,Ho,Hint) );
    }

    cmatFree(M, rho_inv);

}



void imagNL_dOdt (EqDataPkg MC, Cmatrix Orb, Cmatrix dOdt, Cmatrix Ho,
                  Carray Hint, Cmatrix rho1, Carray rho2)
{

/** TIME DERIVATIVE ACCORDING TO NONLINEAR PART
    ===========================================
    Compute essentially the nonlinear part of the equation, multiplied by
    minus imaginary unit. In split-step techniques give us the derivative
    when solving nonlinear part alone. 

    OUTPUT ARGUMENT : dOdt **/

    int
        k,
        s,
        j,
        M = MC->Morb,
        Mpos  = MC->Mpos;



    double
        dx = MC->dx;



    double complex
        Proj,
        g = MC->g;



    Cmatrix
        rho_inv = cmatDef(M, M);



    // Compute inversion and check if it was successfull
    s = HermitianInv(M, rho1, rho_inv);
    if (s != 0)
    {
        printf("\n\n\n\nFailed on Lapack inversion routine!\n");
        printf("-----------------------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(M,M,rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n", s);
        else       printf("\nInvalid argument given : %d\n\n", s);

        exit(EXIT_FAILURE);
    }



    // Update k-th orbital at discretized position j
    #pragma omp parallel for private(k,j) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
            dOdt[k][j] = - I * \
            nonlinear(M, k, j, g, Orb, rho_inv, rho2, Ho, Hint);
    }

    cmatFree(M, rho_inv);
}









/*************************************************************************
 *************************************************************************
 **************                                             **************
 **************    RUNGE-KUTTA INTEGRATOS NONLINEAR PART    **************
 **************                                             **************
 *************************************************************************
 *************************************************************************/

void imagNLTRAP_RK2(EqDataPkg MC, ManyBodyPkg S, doublec dt)
{

/** NonLinear plus TRAP 1-body potential integrator with
    Runge-Kutta of 2nd order for imaginary time
    ====================================================
    Apply 2nd order Runge-Kutta integrator to evolve in time the equation
    corresponding to nonlinear part from split-step with additionally the
    1-body potential to completely separate the derivatives  which  shall
    be evaluated with specral methods. For this purpose 'imagNLTRAP_dOdt'
    is used. The time must be t = - i T.

    OUTPUT :
        S->Orb - Advanced in time step 'dt' using only nonlinear part **/

    int
        i,
        k,
        j,
        Morb,
        Mpos;

    double
        g,
        dx,
        a2,
        * V;

    double complex
        a1;

    Cmatrix
        Ok,
        Oarg;

    Morb = MC->Morb;
    Mpos = MC->Mpos;
    a1 = MC->a1;
    a2 = MC->a2;
    dx = MC->dx;
    g = MC->g;
    V = MC->V;

    Oarg = cmatDef(Morb,Mpos);
    Ok = cmatDef(Morb,Mpos);

    // COMPUTE K1
    imagNLTRAP_dOdt(MC,S->Omat,Ok,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + Ok[k][j] * 0.5 * dt;
        }
    }

    // Update hamiltonian matrix elements
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);

    // COMPUTE K2
    imagNLTRAP_dOdt(MC,Oarg,Ok,S->Ho,S->Hint,S->rho1,S->rho2);

    // Runge-Kutta of 2nd order update
    // y(t+dt) = y(t) + dt * ( f(t + dt/2, y(t) + f(t,y(t))*dt/2) )
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt;
        }
    }

    cmatFree(Morb,Ok);
    cmatFree(Morb,Oarg);

}



void imagNL_RK2(EqDataPkg MC, ManyBodyPkg S, doublec dt)
{

/** NonLinear integrator with Runge-Kutta of 2nd order for imaginary time
    =====================================================================
    Apply 2nd order Runge-Kutta integrator to evolve in time the equation
    corresponding to nonlinear part from split-step without including the
    1-body potential which is taken in account on Finite differences. For
    this purpose 'imagNL_dOdt' is used. The time must be t = - i T.

    OUTPUT :
        S->Orb - Advanced in time step 'dt' using only nonlinear part **/

    int
        i,
        k,
        j,
        Morb,
        Mpos;

    double
        g,
        dx,
        a2,
        * V;

    double complex
        a1;

    Cmatrix
        Ok,
        Oarg;

    Morb = MC->Morb;
    Mpos = MC->Mpos;
    a1 = MC->a1;
    a2 = MC->a2;
    dx = MC->dx;
    g = MC->g;
    V = MC->V;
    
    Oarg = cmatDef(Morb,Mpos);
    Ok = cmatDef(Morb,Mpos);

    // COMPUTE K1
    imagNL_dOdt(MC,S->Omat,Ok,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + Ok[k][j] * 0.5 * dt;
        }
    }

    // Update hamiltonian matrix elements
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);

    // COMPUTE K2
    imagNL_dOdt(MC,Oarg,Ok,S->Ho,S->Hint,S->rho1,S->rho2);

    // Runge-Kutta of 2nd order update
    // y(t+dt) = y(t) + dt * ( f(t + dt/2, y(t) + f(t,y(t))*dt/2) )
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt;
        }
    }

    cmatFree(Morb,Ok);
    cmatFree(Morb,Oarg);

}



void EXPDVRRK2(EqDataPkg MC, ManyBodyPkg S, Carray DerDVR, doublec dt)
{

    int
        k,
        j,
        s,
        Morb,
        Mpos;

    Cmatrix
        Ok,
        Oarg,
        rho1_inv;

    Morb = MC->Morb;
    Mpos = MC->Mpos;

    Oarg = cmatDef(Morb,Mpos);
    Ok = cmatDef(Morb,Mpos);
    rho1_inv = cmatDef(Morb,Morb);

    // INVERSION OF 1-BODY DENSITY MATRIX REQUIRED TO EVALUATE NONLINEAR PART
    s = HermitianInv(Morb,S->rho1,rho1_inv);
    if (s != 0)
    {
        printf("\n\n\nFailed on Lapack inversion routine!\n");
        printf("-----------------------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(Morb,Morb,S->rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n", s);
        else       printf("\nInvalid argument given : %d\n\n", s);

        exit(EXIT_FAILURE);
    }

    // COMPUTE K1
    imagderEXPDVR(MC,S->Omat,Ok,rho1_inv,S->rho2,DerDVR);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + Ok[k][j] * 0.5 * dt;
        }
    }

    // COMPUTE K2
    imagderEXPDVR(MC,Oarg,Ok,rho1_inv,S->rho2,DerDVR);

    // Runge-Kutta of 2nd order update
    // y(t+dt) = y(t) + dt * ( f(t + dt/2, y(t) + f(t,y(t))*dt/2) )
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos-1; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt;
        }
        S->Omat[k][Mpos-1] = S->Omat[k][0];
    }

    cmatFree(Morb,Ok);
    cmatFree(Morb,Oarg);
    cmatFree(Morb,rho1_inv);

}










/***********************************************************************
 ***********************************************************************
 ******************                                    *****************
 ******************       SPLIT-STEP INTEGRATORS       *****************
 ******************                                    *****************
 ***********************************************************************
 ***********************************************************************/

int imagSSFFT(EqDataPkg MC, ManyBodyPkg S, double dT, int Nsteps, int coefInteg)
{

/** INTEGRATE EQUATIONS IN IMAGINARY TIME
    =====================================
    Integrate the MCTDHB equations in imaginary time to find the ground
    state. The integration is done using split-step technique  for  the
    orbitals, where the nonlinear/potential part is  evolved  with  2nd
    order Runge-Kutta scheme and the linear(derivatives)  with spectral
    method using FFTs. The coefficients may be chosen in job.conf  file
    between Lanczos and 4th order Runge-Kutta.

    The many-body data in 'S' is updated until find the ground state **/

    int i,
        j,
        k,
        m,
        l,
        nc,
        Npar,
        Mpos,
        Morb,
        isTrapped;

    MKL_LONG
        p;

    double
        eigFactor,
        freq,
        Idt,
        R2,
        dx,
        a2,
        g;

    Rarray
        V,
        occ;

    double complex
        vir,
        E,
        prevE,
        a1,
        dt;

    Carray
        exp_der;

    DFTI_DESCRIPTOR_HANDLE
        desc;



    // unpack equation parameters
    V = MC->V;
    g = MC->g;
    dx = MC->dx;
    a2 = MC->a2;
    a1 = MC->a1;

    // pure imag. time for relaxation with dT the step size
    dt = - I * dT;
    // (- i * dt) factor found in exponential of split-step
    Idt = - dT;

    // Unpack domain and configuration parameters
    nc = MC->nc;
    Npar = MC->Npar;
    Mpos = MC->Mpos;
    Morb = MC->Morb;
    m = Mpos - 1;
    l = Morb - 1;

    // natural occupations
    occ = rarrDef(Morb);

    // Exponential of derivatives in FFT momentum space. The FFTs ignores
    // the last grid-point assuming periodicity there. Thus the size of
    // functions in FFT must be Mpos - 1
    exp_der = carrDef(m);



    // Check if the system has some trap potential to later verify if
    // the grid domain is not oversized for localized distributions
    isTrapped = strcmp(MC->Vname, "harmonic");



    // setup descriptor (MKL implementation of FFT)
    p = DftiCreateDescriptor(&desc,DFTI_DOUBLE, DFTI_COMPLEX,1,m);
    p = DftiSetValue(desc,DFTI_FORWARD_SCALE,1.0 / sqrt(m));
    p = DftiSetValue(desc,DFTI_BACKWARD_SCALE,1.0 / sqrt(m));
    p = DftiCommitDescriptor(desc);



    // Exponential of derivative operator in momentum space
    for (i = 0; i < m; i++)
    {
        if (i <= (m - 1) / 2) { freq = (2 * PI * i) / (m * dx);       }
        else                  { freq = (2 * PI * (i - m)) / (m * dx); }
        // exponential of derivative operators in half time-step
        exp_der[i] = cexp( -0.5 * dT * (I * a1 * freq - a2 * freq * freq) );
    }



    // Setup one/two-body hamiltonian matrix elements
    SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

    // Setup one/two-body density matrix 
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);



    // Store the initial energy
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    prevE = E;
    vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
    R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
    eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
    hermitianEigvalues(Morb,S->rho1,occ);

    printf("\n\nProgress    E/particle     sqrt<R^2>");
    printf("     |Virial/E|     |H[C]-E*C|    n0");
    sepline();
    printf("%5.1lf%%     %11.7lf",0.0,creal(E));
    printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
    printf("      %8.5lf     %3.0lf%%",eigFactor,100*occ[l]/Npar);



    for (i = 0; i < Nsteps; i++)
    {

        // INTEGRATE ORBITALS
        // Half-step linear part
        linearFFT(Mpos, Morb, &desc, exp_der, S->Omat);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        // Full step nonlinear part
        imagNLTRAP_RK2(MC, S, dt);
        // Half-step linear part again
        linearFFT(Mpos, Morb, &desc, exp_der, S->Omat);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        // INTEGRATE COEFFICIENTS
        if (coefInteg < 2)
        {
            coef_RK4(MC,S,dt);
        }
        else
        {
            LanczosIntegrator(coefInteg,MC,S->Ho,S->Hint,dt,S->C);
        }



        // Loss of Norm => undefined behavior on orthogonality
        Ortonormalize(Morb, Mpos, dx, S->Omat);
        // Renormalize coeficients
        renormalizeVector(nc, S->C, 1.0);



        // Finish time steo calculations updating matrices
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);



        if ( (i + 1) % (Nsteps/5) == 0 && isTrapped == 0)
        {

            // After some time evolved check if initial domain is suitable
            // for the current working orbitals, to avoid oversized domain,
            // a useless length where the functions are zero anyway

            ResizeDomain(MC,S);

            dx = MC->dx;

            Ortonormalize(Morb, Mpos, dx, S->Omat);

            SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
            SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

            // Reconfigure exponential of derivatives in fourier space
            for (j = 0; j < m; j++)
            {
                if (j <= (m-1)/2) { freq = (2 * PI * j) / (m * dx);       }
                else              { freq = (2 * PI * (j - m)) / (m * dx); }
                // exponential of derivative operators in half time-step
                exp_der[j] = cexp(-0.5*dT * (I*a1*freq - a2*freq*freq));
            }

        }



        E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
        vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
        R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
        if ( (i + 1) % (Nsteps / 500) == 0 )
        {
            hermitianEigvalues(Morb,S->rho1,occ);
            eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
            printf("\n%5.1lf%%     %11.7lf",(100.0*i)/Nsteps,creal(E));
            printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
            printf("      %8.5lf     %3.0lf%%",eigFactor,100*occ[l]/Npar);
        }



        // At every 200 time-steps performed
        if ( (i + 1) % 200 == 0)
        {
            // Check if the energy stop decreasing and break the process
            if (fabs(creal(E - prevE) / creal(prevE)) < 5E-10)
            {
                p = DftiFreeDescriptor(&desc);
                free(exp_der);

                // Check if "it is(good enough) eigenstate". In negative
                // case perform a final diagonalization
                eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
                if (eigFactor > 1E-3)
                {
                    if (300 * nc < 5E7)
                    {
                        if (2 * nc / 3 < 300) k = 2 * nc / 3;
                        else                  k = 300;
                    }
                    else k = 5E7 / nc;

                    E = LanczosGround(k,MC,S->Omat,S->C) / Npar;
                }

                OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
                TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,
                      MC->strideTT,MC->IF,S->C,S->rho2);

                E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
                vir = Virial(MC,S->Omat,S->rho1,S->rho2) / Npar;
                R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
                eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);

                printf("\n -END-     %11.7lf",creal(E));
                printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
                printf("      %8.5lf     %3.0lf%%",eigFactor,100*occ[l]/Npar);

                sepline();

                printf("\nProcess ended before because ");
                printf("energy stop decreasing.\n\n");

                return i + 1;
            }

            prevE = E;
        }

    }



    // PROCESS DID NOT STOP AUTOMATICALLY
    // Decide a number of iterations to apprimorate the ground state
    // using Lanczos. Try to avoid memory breakdown
    if (300 * nc < 5E7)
    {
        if (2 * nc / 3 < 300) k = 2 * nc / 3;
        else                  k = 300;
    }
    else k = 5E7 / nc;

    E = LanczosGround(k,MC,S->Omat,S->C) / Npar;
    // Renormalize coeficients
    renormalizeVector(nc,S->C,1.0);

    sepline();
    printf("\nFinal E/particle = %.7lf\n", creal(E));
    printf("Process did not stop automatically because energy were");
    printf(" varying above accepted tolerance\n\n");

    p = DftiFreeDescriptor(&desc);
    free(exp_der);
    free(occ);

    return Nsteps + 1;
}



int imagSSFD(EqDataPkg MC, ManyBodyPkg S, double dT, int Nsteps,
             int coefInteg)
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

    int i,
        j,
        k,
        l,
        nc,
        Npar,
        Mpos,
        Morb,
        isTrapped;

    double
        R2,
        eigFactor,
        dx,
        a2,
        g;

    Rarray
        V,
        occ;

    double complex
        E,
        vir,
        prevE,
        a1,
        dt;

    Carray
        upper,
        lower,
        mid;



    // UNPACK CONFIGURATIONAL DOMAIN PARAMETERS
    nc = MC->nc;
    Npar = MC->Npar;
    Mpos = MC->Mpos;
    Morb = MC->Morb;
    dt = - I * dT;
    dx = MC->dx;

    // UNPACK EQUATION PARAMETERS AND 1-BODY POTENTIAL
    V = MC->V;
    g = MC->g;
    a2 = MC->a2;
    a1 = MC->a1;

    // CONFIGURE TRIDIAGONAL SYSTEM
    upper = carrDef(Mpos-1);
    lower = carrDef(Mpos-1);
    mid = carrDef(Mpos);

    // VECTOR TO MONITOR NATURAL OCCUPATION/CONDENSATION
    l = Morb - 1; // highest occupation index
    occ = rarrDef(Morb);

    // FROM 1-BODY POTENTIAL DECIDE THE TYPE OF BOUNDARIES
    isTrapped = 0;
    if (strcmp(MC->Vname,"harmonic") == 0)      isTrapped = 1;
    if (strcmp(MC->Vname,"doublewell") == 0)    isTrapped = 1;
    if (strcmp(MC->Vname,"harmonicgauss") == 0) isTrapped = 1;

    // SETUP ONE/TWO-BODY HAMILTONIAN MATRIX ELEMENTS
    SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

    // SETUP ONE/TWO-BODY DENSITY MATRIX
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);

    E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
    prevE = E;
    vir = Virial(MC,S->Omat,S->rho1,S->rho2) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
    hermitianEigvalues(Morb,S->rho1,occ);

    // OBSERVABLES FOR QUALITY CONTROL OF THE RESULTS
    printf("\n\nProgress    E/particle     sqrt<R^2>");
    printf("     |Virial/E|     |H[C]-E*C|    n0");
    sepline();
    printf("%5.1lf%%     %11.7lf",0.0,creal(E));
    printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
    printf("      %8.5lf     %3.0lf%%",eigFactor,100*occ[l]/Npar);

    setupTriDiagonal(MC,upper,lower,mid,dt/2,isTrapped);

    for (i = 0; i < Nsteps; i++)
    {

        // PROPAGATE LINEAR PART HALF STEP
        linearCN(MC,upper,lower,mid,S->Omat,isTrapped,dt/2);
        SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
        SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

        // PROPAGATE NONLINEAR PART AN ENTIRE STEP
        imagNL_RK2(MC,S,dt);

        // PROPAGATE LINEAR PART HALF STEP
        linearCN(MC,upper,lower,mid,S->Omat,isTrapped,dt/2);
        SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
        SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);



        // PROPAGATE COEFFICIENTS
        if (coefInteg < 2)
        {
            coef_RK4(MC,S,dt);
        }
        else
        {
            LanczosIntegrator(coefInteg,MC,S->Ho,S->Hint,dt,S->C);
        }



        // Renormalize coeficients
        renormalizeVector(nc, S->C, 1.0);
        // Loss of Norm => undefined behavior on orthogonality
        Ortonormalize(Morb, Mpos, dx, S->Omat);



        // Update quantities that depends on orbitals and coefficients
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);



        if ( (i + 1) % (Nsteps/5) == 0 && isTrapped)
        {
            // After some time evolved check if initial domain is suitable
            // for the current working orbitals, to avoid oversized domain,
            // a useless length where the functions are zero anyway

            ResizeDomain(MC, S);

            dx = MC->dx;

            Ortonormalize(Morb, Mpos, dx, S->Omat);

            SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
            SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

            // re-onfigure again the Crank-Nicolson Finite-difference matrix
            setupTriDiagonal(MC,upper,lower,mid,dt/2,isTrapped);
        }



        E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
        vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
        R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
        if ( (i + 1) % (Nsteps / 500) == 0 )
        {
            eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
            hermitianEigvalues(Morb,S->rho1,occ);
            // Print in screen them on screen
            printf("\n%5.1lf%%     %11.7lf",(100.0*i)/Nsteps,creal(E));
            printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
            printf("      %8.5lf     %3.0lf%%",eigFactor,100*occ[l]/Npar);
        }



        // CHECK IF THE ENERGY HAS STABILIZED TO STOP PROCESS
        // At every 200 time-steps performed
        if ( (i + 1) % 200 == 0)
        {
            // Check if the energy stop decreasing and break the process
            if (fabs(creal(E - prevE) / creal(prevE)) < 5E-10)
            {
                free(upper);
                free(lower);
                free(mid);

                // Check if "it is(good enough) eigenstate". In negative
                // case perform a final diagonalization
                eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
                if (eigFactor > 1E-3)
                {
                    if (300 * nc < 5E7)
                    {
                        if (2 * nc / 3 < 300) k = 2 * nc / 3;
                        else                  k = 300;
                    }
                    else k = 5E7 / nc;

                    E = LanczosGround(k,MC,S->Omat,S->C) / Npar;
                }

                OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
                TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,
                      MC->strideTT,MC->IF,S->C,S->rho2);

                E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
                vir = Virial(MC,S->Omat,S->rho1,S->rho2) / Npar;
                R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
                eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
                hermitianEigvalues(Morb,S->rho1,occ);

                printf("\n -END-     %11.7lf",creal(E));
                printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
                printf("      %8.5lf     %3.0lf%%",eigFactor,100*occ[l]/Npar);

                sepline();

                printf("\nProcess ended before because ");
                printf("energy stop decreasing.\n\n");

                return i + 1;
            }

            prevE = E;
        }

    }

    if (300 * nc < 5E7)
    {
        if (2 * nc / 3 < 300) k = 2 * nc / 3;
        else                  k = 300;
    }
    else k = 5E7 / nc;

    E = LanczosGround(k,MC,S->Omat,S->C) / Npar;
    renormalizeVector(nc, S->C, 1.0);

    sepline();
    printf("\nFinal E/particle = %.7lf\n", creal(E));
    printf("Process did not stop automatically because energy were");
    printf(" varying above accepted tolerance\n\n");

    free(upper);
    free(lower);
    free(mid);
    free(occ);

    return Nsteps + 1;
}



int imagEXPDVR(EqDataPkg MC, ManyBodyPkg S, double dT, int Nsteps,
             int coefInteg)
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

    int i,
        j,
        k,
        l,
        p,
        q,
        N,
        nc,
        Npar,
        Mpos,
        Morb,
        kmom,
        isTrapped;

    double
        eigFactor,
        R2,
        dx,
        a2,
        g,
        L;

    double complex
        E,
        a1,
        dt,
        vir,
        sum,
        diag,
        prevE;

    Rarray
        V,
        occ;

    Carray
        D1DVR,
        D2DVR,
        DerDVR;



    // FROM 1-BODY POTENTIAL DECIDE THE TYPE OF BOUNDARIES
    isTrapped = 0;
    if (strcmp(MC->Vname,"harmonic") == 0)      isTrapped = 1;
    if (strcmp(MC->Vname,"doublewell") == 0)    isTrapped = 1;
    if (strcmp(MC->Vname,"harmonicgauss") == 0) isTrapped = 1;

    // UNPACK DOMAIN PARAMETERS TO LOCAL VARIABLES
    nc = MC->nc;
    Npar = MC->Npar;
    Mpos = MC->Mpos;
    Morb = MC->Morb;
    dx = MC->dx;
    dt = - I*dT;

    // UNPACK EQUATION COEFFICIENTS AND ONE-BODY POTENTIIAL
    a2 = MC->a2;
    a1 = MC->a1;
    g = MC->g;
    V = MC->V;

    // NUMBER OF DVR POINTS IGNORING LAST GRID POINT CONSIDERED AS
    // THE SAME OF THE FIRST ONE - PERIODIC BOUNDARY
    N = Mpos - 1;

    // LENGTH OF GRID DOMAIN
    L = N * dx;

    // REQUEST MEMORY ALLOCATION
    l = Morb - 1;
    occ = rarrDef(Morb);
    D2DVR = carrDef(N * N);  // second derivative matrix in DVR basis
    D1DVR = carrDef(N * N);  // unitary transformation to DVR basis
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
                kmom = k - N/2;
                // NOTE THIS MINUS SIGN - I HAD NO EXPLANATION FOR IT
                diag = - (2*I*PI*kmom/L);
                sum = sum + diag * cexp(2*I*kmom*(j-i)*PI/N) / N;
            }
            D1DVR[i*N + j] = sum;
            D1DVR[j*N + i] = -conj(sum);
        }
        // COMPUTE SEPARATELY THE DIAGONAL
        sum = 0;
        for (k = 0; k < N; k++)
        {
            kmom = k - N/2;
            diag = (2*I*PI*kmom/L);
            sum = sum + diag / N;
        }
        D1DVR[i*N + i] = sum;
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
                kmom = k - N/2;
                diag = - (2*PI*kmom/L)*(2*PI*kmom/L);
                sum = sum + diag * cexp(2*I*kmom*(j-i)*PI/N) / N;
            }
            D2DVR[i*N + j] = sum;
            D2DVR[j*N + i] = conj(sum);
        }
        // COMPUTE SEPARATELY THE DIAGONAL
        sum = 0;
        for (k = 0; k < N; k++)
        {
            kmom = k - N/2;
            diag = - (2*PI*kmom/L)*(2*PI*kmom/L);
            sum = sum + diag / N;
        }
        D2DVR[i*N + i] = sum;
    }

    // SETUP MATRIX CORRESPONDING TO DERIVATIVES ON HAMILTONIAN
    // INCLUDING THE EQUATION COEFFICIENTS IN FRONT OF THEM
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            DerDVR[i*N + j] = a2 * D2DVR[i*N + j] + a1 * D1DVR[i*N + j];
        }
    }

    // SETUP ONE/TWO-BODY HAMILTONIAN MATRIX ELEMENTS ON ORBITAL BASIS
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // SETUP ONE/TWO-BODY DENSITY MATRIX
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);

    E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
    prevE = E;
    vir = Virial(MC,S->Omat,S->rho1,S->rho2) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
    hermitianEigvalues(Morb,S->rho1,occ);

    // OBSERVABLES FOR QUALITY CONTROL OF THE RESULTS
    printf("\n\nProgress    E/particle     sqrt<R^2>");
    printf("     |Virial/E|     |H[C]-E*C|    n0");
    sepline();
    printf("%5.1lf%%     %11.7lf",0.0,creal(E));
    printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
    printf("      %8.5lf     %3.0lf%%",eigFactor,100*occ[l]/Npar);



    // VARIATIONAL ORBITAL EVOLUTION
    for (i = 0; i < Nsteps; i++)
    {

        // FORWARD ONE TIME STEP ORBITALS
        EXPDVRRK2(MC,S,DerDVR,dt);
        SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
        SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);
        // LOSS OF NORM => UNDEFINED BEHAVIOR ON ORTHOGONALITY
        Ortonormalize(Morb,Mpos,dx,S->Omat);

        // PROPAGATE COEFFICIENTS
        if (coefInteg < 2) coef_RK4(MC,S,dt);
        else LanczosIntegrator(coefInteg,MC,S->Ho,S->Hint,dt,S->C);
        // RENORMALIZE COEFICIENTS
        renormalizeVector(nc,S->C,1.0);

        // Update quantities that depends on orbitals and coefficients
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);



        // CHECK IF THE BOUNDARIES ARE STILL GOOD FOR TRAPPED SYSTEMS
        if ( (i + 1) % (Nsteps/5) == 0 && isTrapped) 
        {
            // After some time evolved check if initial domain is suitable
            // for the current working orbitals, to avoid oversized domain,
            // a useless length where the functions are zero anyway
            ResizeDomain(MC,S);
            dx = MC->dx;
            Ortonormalize(Morb,Mpos,dx,S->Omat);
            SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
            SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

            // RE-CONFIGURE DVR MATRICES
            L = N * dx;

            for (p = 0; p < N; p++)
            {
                for (q = 0; q < N; q++)
                {
                    sum = 0;
                    for (k = 0; k < N; k++)
                    {
                        kmom = k - N/2;
                        diag = -(2*I*PI*kmom/L);
                        sum = sum + diag * cexp(2*I*kmom*(q-p)*PI/N)/N;
                    }
                    D1DVR[p*N + q] = sum;
                }
            }

            for (p = 0; p < N; p++)
            {
                for (q = 0; q < N; q++)
                {
                    sum = 0;
                    for (k = 0; k < N; k++)
                    {
                        kmom = k - N/2;
                        diag = - (2*PI*kmom/L)*(2*PI*kmom/L);
                        sum = sum + diag * cexp(2*I*kmom*(q-p)*PI/N)/N;
                    }
                    D2DVR[p*N + q] = sum;
                }
            }

            for (p = 0; p < N; p++)
            {
                for (q = 0; q < N; q++)
                {
                    DerDVR[p*N + q] = a2 * D2DVR[p*N+q] + a1 * D1DVR[p*N+q];
                }
            }
        }



        E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
        vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
        R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
        if ( (i + 1) % (Nsteps / 500) == 0 )
        {
            eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
            hermitianEigvalues(Morb,S->rho1,occ);
            // Print in screen them on screen
            printf("\n%5.1lf%%     %11.7lf",(100.0*i)/Nsteps,creal(E));
            printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
            printf("      %8.5lf     %3.0lf%%",eigFactor,100*occ[l]/Npar);
        }



        // CHECK IF THE ENERGY HAS STABILIZED TO STOP PROCESS
        // At every 200 time-steps performed
        if ( (i + 1) % 200 == 0)
        {
            // Check if the energy stop decreasing and break the process
            if (fabs(creal(E - prevE) / creal(prevE)) < 5E-10)
            {
                free(D1DVR);
                free(D2DVR);
                free(DerDVR);

                // Check if "it is(good enough) eigenstate". In negative
                // case perform a final diagonalization
                eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
                if (eigFactor > 1E-3)
                {
                    if (300 * nc < 5E7)
                    {
                        if (2 * nc / 3 < 300) k = 2 * nc / 3;
                        else                  k = 300;
                    }
                    else k = 5E7 / nc;

                    E = LanczosGround(k,MC,S->Omat,S->C) / Npar;
                }

                OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
                TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,
                      MC->strideTT,MC->IF,S->C,S->rho2);

                E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
                vir = Virial(MC,S->Omat,S->rho1,S->rho2) / Npar;
                R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
                eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
                hermitianEigvalues(Morb,S->rho1,occ);

                printf("\n -END-     %11.7lf",creal(E));
                printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
                printf("      %8.5lf     %3.0lf%%",eigFactor,100*occ[l]/Npar);

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
        if (2 * nc / 3 < 300) k = 2 * nc / 3;
        else                  k = 300;
    }
    else k = 5E7 / nc;

    E = LanczosGround(k,MC,S->Omat,S->C) / Npar;
    renormalizeVector(nc, S->C, 1.0);

    sepline();
    printf("\nFinal E/particle = %.7lf\n", creal(E));
    printf("Process did not stop automatically because energy were");
    printf(" varying above accepted tolerance\n\n");

    free(occ);
    free(D2DVR);
    free(D1DVR);
    free(DerDVR);

    return Nsteps + 1;
}

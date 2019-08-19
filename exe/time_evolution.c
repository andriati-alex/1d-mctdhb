#include <string.h>
#include "integrator_routine.h"
#include "linear_potential.h"





/* ==========================================================================



   *  OBTAIN REAL OR IMAGINARY TIME EVOLUTION SOLUTION OF MCTDHB EQUATIONS  *



   REQUIRED FILES
   --------------------------------------------------------------------------

   (1)  setup/prefix_orb.dat

        Text file with a matrix where the k-th column represent the k-th
        orbital. Thus the rows represent the values of these orbitals in
        discretized positions and each column is then an orbital.

   (2)  setup/prefix_eq.dat

        A text file within the values of equation coefficients organized
        by columns, separeted by spaces:

        Col 1 - second order derivative
        Col 2 - imag part of first order derivative(have no real part)
        Col 3 - interaction strength (g)

        Each line is used as one possible setup to find a ground state

   (3)  setup/prefix_config.dat

        A text file with position/time domain information over  which
        the orbitals and coefficients were generated. The numbers are
        in columns separeted by spaces, in each line as follows:

        Col 1 - # of particles (may vary along lines)
        Col 2 - # of orbitals
        Col 5 - (Mpos) # of slices of size (xf - xi) / Mpos
        Col 3 - x_i
        Col 4 - x_f
        Col 5 - dt (may vary along lines)
        Col 6 - # of time-steps (may vary along lines)

   (4)  job.conf

        A text file containing information about weather time is  imaginary
        or real, the boundary conditions, prefix of input and output files,
        as well as the method employed and number of executions to work out

        The lines starting with # are ignored, treated as comments, to keep
        useful annotations about the execution. Any other character used to
        start a line is interpreted as data to be read  and  recorded. Thus
        lines that do not start with # must be given in the following order

        (4.1) string - 'imag' or 'real' define type of integration
        (4.2) string - Name of linear potential(trap)
        (4.3) boolean int - 1 for periodic and 0 for zero boundary
        (4.4) string - input files prefix
        (4.5) string - output files prefix
        (4.6) positive int - method Id
        (4.7) positive int - Number of jobs to be done
        (4.8) boolean int - 1 to use the same initial condition for all jobs
              0 to adopt progressive initial conditions among executions





   OBSERVATIONS
   --------------------------------------------------------------------------

   The number of jobs must lie in 1 up to the number  of  lines in
   _config.dat and _eq.dat files, each  line  defining  parameters
   to run multiple imaginary propagation  for  different  equation
   parameters set. Even if the file has several lines and  Nstates
   is equal one,  the program will run once for the fisrt line.

   NOTE THAT  # OF ORBITALS AND POSITION DOMAIN DISCRETIZATION ARE
   NOT ALLOWED TO CHANGE BETWEEN LINES.

   If in line the # of particles change, then the file '_coef.dat'
   is then opened again to read the coefficients. The program then
   assume the new vector of coefficients is concatenated to   the
   previous one. For example, if the # of particles change  three
   times among the lines in _conf.dat file, lets say N1 N2 and N3,
   thus is read from the file  NC( N1 , Morb ) elements taken  as
   initial condition, and when it changes to N2 the program  read
   more NC( N2 , Morb ) elements from the file taken again as the
   new initial condition and so on.





   CALL
   --------------------------------------------------------------------------

   ./MCTDHB_time > convergence_log.txt





 * ========================================================================= */





void TimePrint(double t)
{
    
    // format and print time in days / hours / minutes

    int
        tt = (int) t,
        days  = 0,
        hours = 0,
        mins  = 0;

    if ( tt / 86400 > 0 )
    { days  = tt / 86400; tt = tt % 86400; }

    if ( tt / 3600  > 0 )
    { hours = tt / 3600;  tt = tt % 3600;  }

    if ( tt / 60    > 0 )
    { mins  = tt / 60;    tt = tt % 60;    }

    printf("%d day(s) %d hour(s) %d minute(s)", days, hours, mins);
}



void ReachNewLine(FILE * f)
{

    // Read until get new line in a opened file.

    char
        sentinel;

    while (1)
    {
        fscanf(f, "%c", &sentinel);
        if (sentinel == '\n' || sentinel == EOF) return;
    }
}










void SaveConf(FILE * confFileOut, EqDataPkg mc)
{

    fprintf(confFileOut, "%d %d %d ", mc->Npar, mc->Morb, mc->Mpos);

    fprintf(confFileOut, "%.10lf %.10lf ", mc->xi, mc->xf);

    fprintf(confFileOut, "%.15lf %.15lf ", mc->a2, cimag(mc->a1));
    
    fprintf(confFileOut, "%.15lf ", mc->inter);

    fprintf(confFileOut, "%.15lf ", mc->p[0]);
    fprintf(confFileOut, "%.15lf ", mc->p[1]);
    fprintf(confFileOut, "%.15lf", mc->p[2]);

    fprintf(confFileOut, "\n");

}










EqDataPkg SetupData(FILE * paramFile, FILE * confFile, double * dt,
          int * N, char Vname [])
{

    int
        k,
        Mdx,
        Npar,
        Morb;

    double
        xi,
        xf,
        dx,
        a2,
        imag,
        inter;

    double
        p[3];

    double complex
        a1;



    // Setup spatial, time, num of particles and orbitals
    // --------------------------------------------------

    k = fscanf(confFile, "%d %d %d %lf %lf %lf %d",
               &Npar, &Morb, &Mdx, &xi, &xf, dt, N);

    dx = (xf - xi) / Mdx;

    printf("\n\nConfiguration Done\n");

    printf("\n\t# of Particles: %4d", Npar);
    printf("\n\t# of Orbitals:  %4d", Morb);
    printf("\n\t# of possible configurations: %d\n", NC(Npar, Morb));



    // Setup Equation parameters
    // -------------------------

    k = fscanf(paramFile, "%lf %lf %lf %lf %lf %lf",
               &a2, &imag, &inter, &p[0], &p[1], &p[2]);

    a1 = 0 + imag * I;

    return PackEqData(Npar, Morb, Mdx + 1, xi, xf, a2, inter, a1, Vname, p);

}










int main(int argc, char * argv[])
{

    omp_set_num_threads(omp_get_max_threads() / 2);
    mkl_set_num_threads(omp_get_max_threads() / 2);

    /* ==================================================================== *
     *                                                                      *
     *                      VARIABLES AND WHAT THEY DO                      *
     *                                                                      *
     * ==================================================================== */

    int
        i,
        k,
        l,
        s;



    int
        N,      // # of time steps to evolve the system
        Mdx,    // # of divisions in space (# of points - 1)
        Npar,   // # of particles
        Morb,   // # of orbitals
        cyclic, // boundary information
        Nlines, // # of jobs to be executed
        method, // integration method
        resetinit;



    double
        start,      // trigger to measure time
        end,        // trigger to finish time per execution
        time_used,  // total time used
        dx,
        xi,
        xf,    // Domain of orbitals [xi, xf] in steps of dx
        dt,    // time step (both for real and imaginary)  
        real,  // real part of read data from file
        imag,  // imag part of read data from file
        check; // check norm/orthogonality



    Rarray
        x;



    double complex
        checkDiag;



    char
        c, // sentinel character to jump comment lines
        timeinfo,       // 'i' or 'r' for imag/real time evolution
        potname[50],    // Trap/linear potential
        strnum[30],     // conversion of integer to string
        infname[120],   // file name prefix of input data
        outfname[120],  // file name prefix of output data
        fname[120];     // general manipulation to open files by name



    FILE
        * job_file,  // Contains essential information to perform the job
                     // with time info (imag/real), boundary info,  input
                     // and output file names to record  results,  method
                     // number and number of integrations to be done.
        * E_file,    // Output energy values for each integration done.
        * coef_file, // File with initial coefficients data.
        * orb_file,  // initial orbitals data.
        * confFile,  // # of particles/orbitals and domain info.
        * paramFile, // Equation parameters of hamiltonian.
        * confFileOut;



    Carray
        to_int, // auxiliar to compute integration
        vir,    // virial at each time step (should be zero)
        E;      // Energy at each time step evolved



    EqDataPkg
        mc;



    ManyBodyPkg
        S;










    /* ====================================================================
                          CONFIGURE TYPE OF INTEGRATION
       ==================================================================== */

    job_file = fopen("job.conf", "r");

    if (job_file == NULL) // impossible to open file
    {
        printf("\n\n\tERROR: impossible to open file %s\n", "job.conf");
        exit(EXIT_FAILURE);
    }

    i = 1;

    while ( (c  = getc(job_file)) != EOF)
    {

        // jump comment line
        if (c == '#') { ReachNewLine(job_file); continue; }
        else          { fseek(job_file, -1, SEEK_CUR);    }

        switch (i)
        {
            case 1:
                fscanf(job_file, "%s", fname);
                timeinfo = fname[0];
                i = i + 1;
                break;
            case 2:
                fscanf(job_file, "%s", potname);
                i = i + 1;
                break;
            case 3:
                fscanf(job_file, "%d", &cyclic);
                i = i + 1;
                break;
            case 4:
                fscanf(job_file, "%s", infname);
                i = i + 1;
                break;
            case 5:
                fscanf(job_file, "%s", outfname);
                i = i + 1;
                break;
            case 6:
                fscanf(job_file, "%d", &method);
                i = i + 1;
                break;
            case 7:
                fscanf(job_file, "%d", &Nlines);
                i = i + 1;
                break;
            case 8:
                fscanf(job_file, "%d", &resetinit);
                i = i + 1;
                break;
        }

        ReachNewLine(job_file);

    }

    fclose(job_file);

    if (i < 9)
    {
        printf("\nNot enough parameters passed in job.conf file.\n\n");
        exit(EXIT_FAILURE);
    }

    if (timeinfo != 'r' && timeinfo != 'R')
    {
        if (timeinfo != 'i' && timeinfo != 'I')
        {
            printf("\n\n\t!   Invalid integrator identifier   !\n\n");
            exit(EXIT_FAILURE);
        }
    }

    if (method != 1 && method != 2 && method != 3)
    {
        printf("\n\n\tInvalid method Id ! Valid: 1 or 2 or 3\n\n");
        exit(EXIT_FAILURE);
    }





    // Print what it is going to do
    if (timeinfo == 'i' || timeinfo == 'I')
    {
        printf("\n\nIMAGINARY TIME - Method chosen : %d - ", method);
        switch (method)
        {
            case 1:
                printf("Crank-Nicolson SM and ");
                break;
            case 2:
                printf("Crank-Nicolson LU and ");
                break;
            case 3:
                printf("FFT and ");
                break;
        }

        printf("4th Order Runge-Kutta");
    }

    else
    {
        printf("\n\nREAL TIME INTEGRATION\n\n");
    }










    /* ====================================================================
                   CONFIGURE TRAP POTENTIAL AND DISCRETIZATION
       ==================================================================== */

    printf("\n\n\n");
    printf("=========================================================\n\n");
    printf("Using %s setup files to configure integrator\n\n", infname);
    printf("=========================================================\n\n");









    
    /* ====================================================================
                         OPEN FILES TO SETUP THE PROBLEM
       ==================================================================== */

    strcpy(fname, "input/");
    strcat(fname, infname);
    strcat(fname, "_conf.dat");

    printf("\n\nLooking for %s", fname);

    confFile = fopen(fname, "r");

    if (confFile == NULL) // impossible to open file
    {
        printf("\n\n\tERROR: impossible to open file %s\n", fname);
        exit(EXIT_FAILURE);
    } else
    {
        printf(" .... Found !\n");
    }



    strcpy(fname, "input/");
    strcat(fname, infname);
    strcat(fname, "_eq.dat");

    printf("\nLooking for %s", fname);

    paramFile = fopen(fname, "r");

    if (paramFile == NULL) // impossible to open file
    {
        printf("\n\n\tERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    } else
    {
        printf(" ...... Found !\n");
    }



    strcpy(fname, "input/");
    strcat(fname, infname);
    strcat(fname, "_orb.dat");

    printf("\nLooking for %s ", fname);

    orb_file = fopen(fname, "r");

    if (orb_file == NULL)  // impossible to open file
    {
        printf("\n\n\tERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    } else
    {
        printf(" .... Found !\n");
    }



    strcpy(fname, "input/");
    strcat(fname, infname);
    strcat(fname, "_coef.dat");

    printf("\nLooking for %s ", fname);

    coef_file = fopen(fname, "r");

    if (coef_file == NULL)  // impossible to open file
    {
        printf("\n\n\tERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    } else
    {
        printf(" ... Found !\n");
    }



    /* ====================================================================
                                 OPEN OUTPUT FILES
       ==================================================================== */

    if (timeinfo == 'i' || timeinfo == 'I')
    {

        strcpy(fname, "output/");
        strcat(fname, outfname);
        strcat(fname, "_energy_imagtime.dat");

        E_file = fopen(fname, "w");

        if (E_file == NULL)  // impossible to open file
        {
            printf("\n\n\tERROR: impossible to open file %s\n\n", fname);
            exit(EXIT_FAILURE);
        }
    }

    strcpy(fname, "output/");
    strcat(fname, outfname);
    strcat(fname, "_conf.dat");

    confFileOut = fopen(fname, "w");
    if (confFileOut == NULL) // impossible to open file
    {
        printf("\n\n\tERROR: impossible to open file %s\n", fname);
        exit(EXIT_FAILURE);
    }

    fprintf(confFileOut, "# trap : %s\n", potname);










    /* ====================================================================
           READ DATA TO SETUP EQUATION PARAMETERS AND INITIAL CONDITIONS
       ==================================================================== */

    mc = SetupData(paramFile, confFile, &dt, &N, potname);

    dx = mc->dx;
    xi = mc->xi;
    xf = mc->xf;
    Mdx = mc->Mpos - 1;
    Npar = mc->Npar;
    Morb = mc->Morb;

    x = rarrDef(Mdx + 1);
    rarrFillInc(Mdx + 1, xi, dx, x);



    S = AllocManyBodyPkg(Npar,Morb,Mdx + 1);

    // Setup orbitals
    // --------------

    for (k = 0; k < Mdx + 1; k++)
    {
        for (s = 0; s < Morb; s++)
        {
            l = fscanf(orb_file, " (%lf%lfj) ", &real, &imag);
            S->Omat[s][k] = real + I * imag;
        }
    }

    Ortonormalize(Morb, Mdx + 1, dx, S->Omat);

    // orbitals are not read again
    fclose(orb_file);



    // Setup Coeficients
    // -----------------

    for (k = 0; k < NC(Npar, Morb); k++)
    {
        l = fscanf(coef_file, " (%lf%lfj)", &real, &imag);
        S->C[k] = real + I * imag;
    }










    /* ====================================================================
                       CHECK ORTHOGONALITY AND NORMALIZATION
       ==================================================================== */

    printf("\n\n\n");
    printf("=========================================================\n\n");
    printf("Configuration done. Checking orthonormality\n\n");
    printf("=========================================================\n\n");


    to_int = carrDef(Mdx + 1);



    // Check if off-diagonal elements are zero
    // ---------------------------------------

    check = 0;
    for (k = 0; k < Morb; k++)
    {
        for (l = 0; l < Morb; l++)
        {
            if (l == k) continue;
            for (s = 0; s < Mdx + 1; s++)
            {
                to_int[s] = conj(S->Omat[k][s]) * S->Omat[l][s];
            }
            check = check + cabs(Csimps(Mdx + 1, to_int, mc->dx));
        }
    }

    if (check > 1E-8)
    {
        printf("\n\n\t!   ORBITALS ARE NOT ORTHOGONAL   !\n");
        return -1;
    }



    // Check if Diagonal elements sum up to Morb
    // -----------------------------------------

    checkDiag = 0;
    for (k = 0; k < Morb; k++)
    {
        for (s = 0; s < Mdx + 1; s++)
        {
            to_int[s] = conj(S->Omat[k][s]) * S->Omat[k][s];
        }
        checkDiag = checkDiag + Csimps(Mdx + 1, to_int, mc->dx);
    }

    if (abs(creal(checkDiag) - Morb) > 1E-8 || cimag(checkDiag) > 1E-8)
    {
        printf("\n\n\t!   ORBITALS DO NOT HAVE NORM = 1   !\n");
        return -1;
    }



    // Check normalization of coeficients
    // ----------------------------------

    if ( abs(carrMod2(NC(Npar, Morb), S->C) - 1) > 1E-9 )
    {
        printf("\n\n\t!   COEFFICIENTS DO NOT HAVE NORM = 1   !\n");
        return -1;
    }










    /* ====================================================================
                               REAL TIME INTEGRATION
       ==================================================================== */

    if (timeinfo == 'r' || timeinfo =='R')
    {
        printf("\n\n\n");
        printf("=======================================================\n\n");
        printf("Start real time Integration\n\n");
        printf("=======================================================\n\n");

        fclose(confFile);
        fclose(paramFile);
        fclose(coef_file);

        start = omp_get_wtime();

        REAL_FP(mc, S, dt, N, cyclic, outfname, 1);

        time_used = (double) (omp_get_wtime() - start);

        printf("\n\nTime taken in real time integration : %lf sec",time_used);
        printf(" = "); TimePrint(time_used);

        SaveConf(confFileOut, mc);

        // Record Trap potential

        strcpy(fname, "output/");
        strcat(fname, outfname);
        strcat(fname, "_trap.dat");

        rarr_txt(fname, Mdx + 1, mc->V);

        ReleaseEqDataPkg(mc);
        ReleaseManyBodyDataPkg(S);
        free(to_int);
        free(x);

        fclose(confFileOut);

        printf("\n\n");
        return 0;

    }










    /* ====================================================================
             DIAGONALIZE HAMILTONIAN IN THE GIVEN BASIS BEFORE START
       ==================================================================== */
    
    // Estimate energy by diagonalization Restrict the number of iterations
    // in lanczos routine to avoid massive memory usage. Try to use 200
    // iterations unless either it exceeds half of the dimension of
    // configuration space or if it exceeds a memory Threshold.

    E   = carrDef(N + 1); // to store energy
    vir = carrDef(N + 1); // check consistency by Virial Theorem
    
    if ( dt > 5 * dx * dx && method < 3)
    {
        printf("\n\nWARNING : time-step too large to maintain stability");
        printf(" in finite-differences methods.\n\n");
    }

    if (200 * NC(Npar, Morb) < 5E7)
    {
        if (NC(Npar, Morb) / 2 < 200) k = NC(Npar, Morb) / 2;
        else                          k = 200;
    }
    else k = 5E7 / NC(Npar, Morb);

    E[0] = LanczosGround( k, mc, S->Omat, S->C );
    // Renormalize coeficients
    renormalizeVector(NC(Npar, Morb), S->C, 1.0);

    printf("\n=====================================================");
    printf("==========================\n\n");
    printf("\tDiagonalization Done. Initial energy = %.5E", creal(E[0]));
    printf("\n\t# of lanczos iterations used = %d", k);
    printf("\n\n===================================================");
    printf("============================");










    /* ====================================================================
                                CALL THE INTEGRATOR 
     * ==================================================================== */

    printf("\n\n\n");
    printf("=========================================================\n\n");
    printf("Start imaginary time Integration #%d\n\n", 1);
    printf("=========================================================\n\n");

    switch (method)
    {

        case 1:

            start = omp_get_wtime();
            s = IMAG_RK4_CNSMRK4(mc, S, E, vir, dt, N, cyclic);
            time_used = (double) (omp_get_wtime() - start);
            printf("\n\nTime taken in integration #%d : %lf", 1, time_used);
            printf(" = "); TimePrint(time_used);

            break;

        case 2:

            start = omp_get_wtime();
            s = IMAG_RK4_CNLURK4(mc, S, E, vir, dt, N, cyclic);
            time_used = (double) (omp_get_wtime() - start);
            printf("\n\nTime taken in integration #%d : %lf", 1, time_used);
            printf(" = "); TimePrint(time_used);

            break;

        case 3:

            start = omp_get_wtime();
            s = IMAG_RK4_FFTRK4(mc, S, E, vir, dt, N);
            time_used = (double) (omp_get_wtime() - start);
            printf("\n\nTime taken in integration #%d : %lf", 1, time_used);
            printf(" = "); TimePrint(time_used);

            break;
    }



    // Record data
    // ---------------------------------------------

    // setup filename to record solution
    strcpy(fname, "output/");
    strcat(fname, outfname);
    strcat(fname, "_line-1");

    strcat(fname, "_orb_imagtime.dat");
    cmat_txt_T(fname, Morb, Mdx + 1, S->Omat);

    // Record Coeficients Data

    strcpy(fname, "output/");
    strcat(fname, outfname);
    strcat(fname, "_line-1");
    strcat(fname, "_coef_imagtime.dat");

    carr_txt(fname, mc->nc, S->C);

    // Record Trap potential

    strcpy(fname, "output/");
    strcat(fname, outfname);
    strcat(fname, "_line-1");
    strcat(fname, "_trap.dat");

    rarr_txt(fname, Mdx + 1, mc->V);

    fprintf(E_file, "%.10E\n", creal(E[s-1]));

    SaveConf(confFileOut, mc);

    // update domain that might have changed
    xi = mc->xi;
    xf = mc->xf;
    dx = mc->dx;

    rarrFillInc(Mdx + 1, xi, dx, x);










/** If either the _conf.dat or _eq.dat file have more  than  one  line
    and the Nlines parameter is greater than 1 it read the next config
    from files to perform another time integration
**/

    for (i = 1; i < Nlines; i++)
    {

        printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
        printf("***************************************");
        printf("***************************************");
        printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

        // number of line reading in _conf.dat and _eq.dat files
        sprintf(strnum, "%d", i + 1);

        // release old data
        ReleaseEqDataPkg(mc);

        // setup new parameters
        mc = SetupData(paramFile, confFile, &dt, &N, potname);

        free(E);
        free(vir);

        E   = carrDef(N + 1); // to store energy
        vir = carrDef(N + 1); // check consistency by Virial Theorem

        if (Npar != mc->Npar)
        {

        //  Setup Coeficients from file because the number
        //  of particles has changed

            free(S->C);

            Npar = mc->Npar;
            S->Npar = Npar;

            S->C = carrDef(NC(Npar, Morb));

            for (k = 0; k < NC(Npar, Morb); k++)
            {
                l = fscanf(coef_file, " (%lf%lfj)", &real, &imag);
                S->C[k] = real + I * imag;
            }
        } else
        {

            // The number of particles has not changed Although it will
            // read again if is required to reset initial conditions.

            if (resetinit)
            {
                fclose(coef_file);

                strcpy(fname, "input/");
                strcat(fname, infname);
                strcat(fname, "_coef.dat");

                coef_file = fopen(fname, "r");

                for (k = 0; k < NC(Npar, Morb); k++)
                {
                    l = fscanf(coef_file, " (%lf%lfj)", &real, &imag);
                    S->C[k] = real + I * imag;
                }
            }
        }

        // If restart = True(1) use the same initial orbitals for all jobs
        if (resetinit)
        {
            strcpy(fname, "input/");
            strcat(fname, infname);
            strcat(fname, "_orb.dat");

            printf("\nReseted initial conditions.\n");

            orb_file = fopen(fname, "r");

            for (k = 0; k < Mdx + 1; k++)
            {
                for (s = 0; s < Morb; s++)
                {
                    l = fscanf(orb_file, " (%lf%lfj) ", &real, &imag);
                    S->Omat[s][k] = real + I * imag;
                }
            }

            // orbitals are not read again
            fclose(orb_file);

            // Take configurations from file again to re-setup domain
            xi = mc->xi;
            xf = mc->xf;
            dx = mc->dx;
            rarrFillInc(Mdx + 1, xi, dx, x);

            Ortonormalize(Morb, Mdx + 1, dx, S->Omat);
        }

        else
        {
            mc->xi = xi;
            mc->xf = xf;
            mc->dx = dx;

            GetPotential(Mdx+1,mc->Vname,x,mc->V,mc->p[0],mc->p[1],mc->p[2]);
        }



        /* ================================================================
                         CHECK ORTHOGONALITY AND NORMALIZATION
           ================================================================ */

        printf("\n\n\n");
        printf("=========================================================\n\n");
        printf("Configuration done. Checking orthonormality\n\n");
        printf("=========================================================\n\n");

        // Check if off-diagonal elements are zero

        check = 0;
        for (k = 0; k < Morb; k++)
        {
            for (l = 0; l < Morb; l++)
            {
                if (l == k) continue;
                for (s = 0; s < Mdx + 1; s++)
                {
                    to_int[s] = conj(S->Omat[k][s]) * S->Omat[l][s];
                }
                check = check + cabs(Csimps(Mdx + 1, to_int, mc->dx));
            }
        }

        if (check > 1E-9)
        {
            printf("\n\n\t!   ORBITALS ARE NOT ORTHOGONAL   !\n");
            return -1;
        }

        // Check if Diagonal elements sum up to Morb

        checkDiag = 0;
        for (k = 0; k < Morb; k++)
        {
            for (s = 0; s < Mdx + 1; s++)
            {
                to_int[s] = conj(S->Omat[k][s]) * S->Omat[k][s];
            }
            checkDiag = checkDiag + Csimps(Mdx + 1, to_int, mc->dx);
        }

        if (abs(creal(checkDiag) - Morb) > 1E-8 || cimag(checkDiag) > 1E-8)
        {
            printf("\n\n\t!   ORBITALS DO NOT HAVE NORM = 1   !\n");
            return -1;
        }

        // Check normalization of coeficients

        if ( abs(carrMod2(NC(Npar, Morb), S->C) - 1) > 1E-9 )
        {
            printf("\n\n\t!   COEFFICIENTS DO NOT HAVE NORM = 1   !\n");
            return -1;
        }



        /* =================================================================
                DIAGONALIZE HAMILTONIAN IN THE GIVEN BASIS BEFORE START
           ================================================================= */

        if ( dt > 5 * dx * dx && method < 3)
        {
            printf("\n\nWARNING : time-step too large to maintain stability");
            printf(" in finite-differences methods.\n\n");
        }

        if (200 * NC(Npar, Morb) < 5E7)
        {
            if (NC(Npar, Morb) / 2 < 200) k = NC(Npar, Morb) / 2;
            else                          k = 200;
        }
        else k = 5E7 / NC(Npar, Morb);

        E[0] = LanczosGround(k,mc,S->Omat,S->C);
        // Renormalize coeficients
        renormalizeVector(NC(Npar, Morb), S->C, 1.0);

        printf("\n=====================================================");
        printf("==========================\n\n");
        printf("\tDiagonalization Done E = %.5E", creal(E[0]));
        printf("\n\t# of lanczos iterations used = %d", k);
        printf("\n\n===================================================");
        printf("============================");



        /* ================================================================
                                 CALL THE INTEGRATOR 
         * ================================================================ */

        printf("\n\n\n");
        printf("=======================================================\n\n");
        printf("Start imaginary time Integration #%d\n\n", i + 1);
        printf("=======================================================\n\n");

        switch (method)
        {

            case 1:

                start = omp_get_wtime();
                s = IMAG_RK4_CNSMRK4(mc, S, E, vir, dt, N, cyclic);
                end = (double) (omp_get_wtime() - start);
                time_used += end;
                printf("\n\nTime taken in execution #%d : %.1lf", i+1, end);
                printf(" = "); TimePrint(end);

                break;

            case 2:

                start = omp_get_wtime();
                s = IMAG_RK4_CNLURK4(mc, S, E, vir, dt, N, cyclic);
                end = (double) (omp_get_wtime() - start);
                time_used += end;
                printf("\n\nTime taken in execution #%d : %.1lf", i+1, end);
                printf(" = "); TimePrint(end);

                break;

            case 3:

                start = omp_get_wtime();
                s = IMAG_RK4_FFTRK4(mc, S, E, vir, dt, N);
                end = (double) (omp_get_wtime() - start);
                time_used += end;
                printf("\n\nTime taken in execution #%d : %.1lf", i+1, end);
                printf(" = "); TimePrint(end);

                break;
        }



        // Record data
        // ---------------------------------------------

        // setup filename to store solution
        strcpy(fname, "output/");
        strcat(fname, outfname);
        strcat(fname, "_line-");
        strcat(fname, strnum);

        strcat(fname, "_orb_imagtime.dat");
        cmat_txt_T(fname, Morb, Mdx + 1, S->Omat);

        // Record Coeficients Data

        strcpy(fname, "output/");
        strcat(fname, outfname);
        strcat(fname, "_line-");
        strcat(fname, strnum);
        strcat(fname, "_coef_imagtime.dat");

        carr_txt(fname, mc->nc, S->C);

        // Record trap potential

        strcpy(fname, "output/");
        strcat(fname, outfname);
        strcat(fname, "_line-");
        strcat(fname, strnum);
        strcat(fname, "_trap.dat");

        rarr_txt(fname, Mdx + 1, mc->V);

        fprintf(E_file, "%.10E\n", creal(E[s-1]));

        SaveConf(confFileOut, mc);

    }





    /* ====================================================================
                                  RELEASE MEMORY
     * ==================================================================== */

    fclose(E_file);
    fclose(confFile);
    fclose(paramFile);
    fclose(coef_file);
    fclose(confFileOut);

    ReleaseEqDataPkg(mc);
    ReleaseManyBodyDataPkg(S);
    free(E);
    free(x);
    free(vir);
    free(to_int);

    printf("\n\nTotal time taken: %.1lf = ", time_used);
    TimePrint(time_used);
    
    printf("\n\nAverage time per state: %.1lf = ", time_used / Nlines);
    TimePrint(time_used / Nlines);

    printf("\n\n");
    return 0;
}

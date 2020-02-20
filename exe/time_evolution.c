#include "integrators.h"
#include "linear_potential.h"
#include "structureSetup.h"





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
    
/** format and print time in days / hours / minutes **/

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

/** Read until get new line in a opened file. **/

    char
        sentinel;

    while (1)
    {
        fscanf(f, "%c", &sentinel);
        if (sentinel == '\n' || sentinel == EOF) return;
    }
}



void orthoCheck(int Npar, int Norb, int Ngrid, double dx, Cmatrix Omat,
                Carray C)
{
    int
        s,
        k,
        l;

    double
        overlap;

    Carray
        Integ;

    Integ = carrDef(Ngrid);

    printf("\n\nChecking orthonormality ... ");

    // Check if off-diagonal elements are zero
    overlap = 0;
    for (k = 0; k < Norb; k++)
    {
        for (l = 0; l < Norb; l++)
        {
            if (l == k) continue;
            for (s = 0; s < Ngrid; s++)
            {
                Integ[s] = conj(Omat[k][s])*Omat[l][s];
            }
            overlap = overlap + cabs(Csimps(Ngrid,Integ,dx));
        }
    }

    if (overlap > 1E-8)
    {
        printf("\n\n!   ORBITALS ARE NOT ORTHOGONAL   !\n\n");
        exit(EXIT_FAILURE);
    }

    // Check if norm of all orbitals
    overlap = 0;
    for (k = 0; k < Norb; k++)
    {
        for (s = 0; s < Ngrid; s++)
        {
            Integ[s] = conj(Omat[k][s]) * Omat[k][s];
        }
        overlap = overlap + cabs(Csimps(Ngrid,Integ,dx));
    }

    if (fabs(overlap - Norb) > 1E-8)
    {
        printf("\n\n!   ORBITALS DO NOT HAVE NORM = 1   !\n\n");
        exit(EXIT_FAILURE);
    }

    // Check normalization of coeficients
    if ( abs(carrMod2(NC(Npar,Norb),C) - 1) > 1E-9 )
    {
        printf("\n\n!   COEFFICIENTS DO NOT HAVE NORM = 1   !\n\n");
        exit(EXIT_FAILURE);
    }

    // Everything ok
    printf("Done\n");
    free(Integ);
}



void initDiag(EqDataPkg mc, ManyBodyPkg S)
{
    int
        Lit,
        Npar,
        Norb;

    double complex
        E0;

    Npar = mc->Npar;
    Norb = mc->Morb;

    printf("\nLanczos ground state with initial orbitals");
    printf(" ... ");

    // select a suitable number of lanczos iterations
    if (200 * NC(Npar,Norb) < 5E7)
    {
        if (NC(Npar,Norb) / 2 < 200) Lit = NC(Npar,Norb) / 2;
        else                         Lit = 200;
    }
    else Lit = 5E7 / NC(Npar,Norb);

    E0 = LanczosGround(Lit,mc,S->Omat,S->C);

    printf("Done.\nInitial E0/Npar = %.7lf\n",creal(E0)/Npar);
}



void SaveConf(FILE * confFileOut, EqDataPkg mc)
{

/** Record grid and parameters specifications used **/

    fprintf(confFileOut, "%d %d %d ", mc->Npar, mc->Morb, mc->Mpos);

    fprintf(confFileOut, "%.10lf %.10lf ", mc->xi, mc->xf);

    fprintf(confFileOut, "%.15lf %.15lf ", mc->a2, cimag(mc->a1));
    
    fprintf(confFileOut, "%.15lf ", mc->g);

    fprintf(confFileOut, "%.15lf ", mc->p[0]);
    fprintf(confFileOut, "%.15lf ", mc->p[1]);
    fprintf(confFileOut, "%.15lf", mc->p[2]);

    fprintf(confFileOut, "\n");

}



EqDataPkg SetupData(FILE * paramFile, FILE * confFile, double * dt,
          int * N, char Vname [])
{

/** Read from file input data and return in structure **/

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
    k = fscanf(confFile, "%d %d %d %lf %lf %lf %d",
               &Npar, &Morb, &Mdx, &xi, &xf, dt, N);
    dx = (xf - xi) / Mdx;

    // Setup Equation parameters
    k = fscanf(paramFile, "%lf %lf %lf %lf %lf %lf",
               &a2, &imag, &inter, &p[0], &p[1], &p[2]);

    a1 = 0 + imag * I;

    return PackEqData(Npar,Morb,Mdx+1,xi,xf,a2,inter,a1,Vname,p);

}



int main(int argc, char * argv[])
{

    omp_set_num_threads(omp_get_max_threads() / 2);
    mkl_set_num_threads(omp_get_max_threads() / 2);

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
        coefInteg,
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
        E0;

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
        printf("\n\nERROR: impossible to open file %s\n\n","job.conf");
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
                fscanf(job_file,"%s", fname);
                timeinfo = fname[0];
                i = i + 1;
                break;
            case 2:
                fscanf(job_file,"%s", potname);
                i = i + 1;
                break;
            case 3:
                fscanf(job_file,"%s", infname);
                i = i + 1;
                break;
            case 4:
                fscanf(job_file,"%s", outfname);
                i = i + 1;
                break;
            case 5:
                fscanf(job_file,"%d", &method);
                i = i + 1;
                break;
            case 6:
                fscanf(job_file,"%d", &coefInteg);
                i = i + 1;
                break;
            case 7:
                fscanf(job_file,"%d", &Nlines);
                i = i + 1;
                break;
            case 8:
                fscanf(job_file,"%d", &resetinit);
                i = i + 1;
                break;
        }

        ReachNewLine(job_file);

    }

    fclose(job_file);
    cyclic = 1;





    /* ====================================================================
                CHECK IF THERE ARE ERRORS IN JOB.CONF FILE ENTRIES
       ==================================================================== */

    if (i < 9)
    {
        printf("\n\nERROR : Not enough parameters found in job file.\n");
        printf("Could get %d but expected 9.\n\n",i);
        exit(EXIT_FAILURE);
    }

    if (timeinfo != 'r' && timeinfo != 'R')
    {
        if (timeinfo != 'i' && timeinfo != 'I')
        {
            printf("\n\nERROR : Invalid integrator identifier! ");
            printf("Must be 'imag' or 'real'\n\n");
            exit(EXIT_FAILURE);
        }
    }

    if (method != 1 && method != 2 && method != 3)
    {
        printf("\n\nERROR : Invalid method Id(Orbitals)! Valid ones are:\n");
        printf("\t1 - Crank-Niconsol-SM-RK2\n");
        printf("\t2 - Crank-Niconsol-LU-RK2\n");
        printf("\t3 - Crank-Niconsol-FFT-RK2\n\n\n");
        exit(EXIT_FAILURE);
    }

    if (coefInteg > 5)
    {
        printf("\n\nERROR : Invalid Coef. integrator Id! Valid ones are:\n");
        printf("\t0/1 - 4th order Rung-Kutta\n");
        printf("\t2/3/4/5 - Lanczos number of iterations\n\n\n");
        exit(EXIT_FAILURE);
    }





    printf("\n\n\n");
    printf("\t\t****************************************************\n");
    printf("\t\t*                                                  *\n");
    printf("\t\t*   MCTDHB program initiated                       *\n");
    printf("\t\t*   Developer contact: andriati@if.usp.br          *\n");
    printf("\t\t*                                                  *\n");
    printf("\t\t*                                                  *\n");

    if (timeinfo == 'i' || timeinfo == 'I')
    {
    printf("\t\t*            IMAGINARY TIME PROPAGATION            *\n");
    }
    else
    {
    printf("\t\t*               REAL TIME PROPAGATION              *\n");
    }
    printf("\t\t*                                                  *\n");
    printf("\t\t*                                                  *\n");
    printf("\t\t****************************************************\n");
    printf("\n\n");









    
    /* ====================================================================
                         OPEN FILES TO SETUP THE PROBLEM
       ==================================================================== */

    printf("OPENNING INPUT FILES\n");

    strcpy(fname, "input/");
    strcat(fname, infname);
    strcat(fname, "_conf.dat");

    printf("Looking for %s", fname);

    confFile = fopen(fname, "r");
    if (confFile == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    } else
    {
        printf(" .... Found !\n");
    }



    strcpy(fname, "input/");
    strcat(fname, infname);
    strcat(fname, "_eq.dat");

    printf("Looking for %s", fname);

    paramFile = fopen(fname, "r");
    if (paramFile == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    } else
    {
        printf(" ...... Found !\n");
    }



    strcpy(fname, "input/");
    strcat(fname, infname);
    strcat(fname, "_orb.dat");

    printf("Looking for %s ", fname);

    orb_file = fopen(fname, "r");
    if (orb_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    } else
    {
        printf(" .... Found !\n");
    }



    strcpy(fname, "input/");
    strcat(fname, infname);
    strcat(fname, "_coef.dat");

    printf("Looking for %s ", fname);

    coef_file = fopen(fname, "r");
    if (coef_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
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
            printf("\n\nERROR: impossible to open file %s\n\n", fname);
            exit(EXIT_FAILURE);
        }
    }

    strcpy(fname, "output/");
    strcat(fname, outfname);
    strcat(fname, "_conf.dat");

    confFileOut = fopen(fname, "w");
    if (confFileOut == NULL) // impossible to open file
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }

    fprintf(confFileOut, "# trap Id : %s\n", potname);





    /* ====================================================================
           READ DATA TO SETUP EQUATION PARAMETERS AND INITIAL CONDITIONS
       ==================================================================== */

    printf("\n\n\n\n\n\n\n\n\n\n\n\n");
    printf("******************************      ");
    printf("JOB 1");
    printf("      ******************************\n\n");

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
    for (k = 0; k < Mdx + 1; k++)
    {
        for (s = 0; s < Morb; s++)
        {
            l = fscanf(orb_file, " (%lf%lfj) ", &real, &imag);
            S->Omat[s][k] = real + I * imag;
        }
    }
    // Ortonormalize(Morb, Mdx + 1, dx, S->Omat);
    fclose(orb_file);

    // Setup Coeficients
    for (k = 0; k < NC(Npar, Morb); k++)
    {
        l = fscanf(coef_file, " (%lf%lfj)", &real, &imag);
        S->C[k] = real + I * imag;
    }

    printf("Configuration successfully set up\n");
    printf("=================================");
    printf("\n# of Particles: %4d",Npar);
    printf("\n# of Orbitals:  %4d",Morb);
    printf("\n# of possible configurations: %d",NC(Npar,Morb));
    printf("\n# of grid points: %d",Mdx+1);
    printf("\nDomain boundaries: [%.2lf,%.2lf]",xi,xf);
    printf("\nGrid step: %.2lf",dx);
    printf("\nFinal time : %.1lf in steps of %.6lf",N*dt,dt);
    printf("\nIntegration method : ");
    if (method < 3)
    {
        printf("Crank-Nicolson with Runge-Kutta for Orbitals / ");
    }
    else
    {
        printf("FFT with Runge-Kutta for Orbitals / ");
    }
    if (coefInteg < 2)
    {
        printf("RK4 for coeff.");
    }
    else
    {
        printf("Lanczos for coeff.");
    }

    // Orthonormality check
    orthoCheck(Npar,Morb,Mdx+1,dx,S->Omat,S->C);










    /* ====================================================================
                               REAL TIME INTEGRATION
       ==================================================================== */

    if (timeinfo == 'r' || timeinfo =='R')
    {
        printf("\nStart real time Integration\n");

        fclose(confFile);
        fclose(paramFile);
        fclose(coef_file);

        switch (method)
        {
            case 1:
                start = omp_get_wtime();
                realCNSM(mc,S,dt,N,cyclic,outfname,N/Nlines);
                time_used = (double) (omp_get_wtime() - start);
                break;
            case 2:
                start = omp_get_wtime();
                realCNSM(mc,S,dt,N,cyclic,outfname,N/Nlines);
                time_used = (double) (omp_get_wtime() - start);
                break;
            case 3:
                start = omp_get_wtime();
                realFFT(mc,S,dt,N,outfname,N/Nlines);
                time_used = (double) (omp_get_wtime() - start);
                break;
        }

        printf("\nTime taken in integration : %lf(s) = ",time_used);
        TimePrint(time_used);
        printf("\nAverage per time steps : %.1lf(ms)",1000*time_used/N);

        SaveConf(confFileOut, mc);

        // Record Trap potential

        strcpy(fname, "output/");
        strcat(fname, outfname);
        strcat(fname, "_trap.dat");

        rarr_txt(fname, Mdx + 1, mc->V);

        ReleaseEqDataPkg(mc);
        ReleaseManyBodyDataPkg(S);
        free(x);

        fclose(confFileOut);

        printf("\n\n-- END --\n\n");
        return 0;
    }










    // Diagonalization with initial orbitals to improve coefficients
    // for imaginary time propagation to find ground state
    initDiag(mc,S);
    printf("\nStart imaginary time Integration");

    switch (method)
    {

        case 1:

            start = omp_get_wtime();
            s = imagCNSM(mc, S, dt, N, coefInteg, cyclic);
            time_used = (double) (omp_get_wtime() - start);
            break;

        case 2:

            start = omp_get_wtime();
            s = imagCNLU(mc, S, dt, N, coefInteg, cyclic);
            time_used = (double) (omp_get_wtime() - start);
            break;

        case 3:
            start = omp_get_wtime();
            s = imagFFT(mc, S, dt, N, coefInteg);
            time_used = (double) (omp_get_wtime() - start);
            break;
    }

    printf("Time taken in job%d : %.1lf(s) = ",1,time_used);
    TimePrint(time_used);
    printf("\nAverage per time steps : %.1lf(ms)",time_used/s*1000);

    // Record data

    E0 = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint);

    // setup filename to record solution
    strcpy(fname, "output/");
    strcat(fname, outfname);
    strcat(fname, "_job1");

    strcat(fname, "_orb_imagtime.dat");
    cmat_txt_T(fname, Morb, Mdx + 1, S->Omat);

    // Record Coeficients Data

    strcpy(fname, "output/");
    strcat(fname, outfname);
    strcat(fname, "_job1");
    strcat(fname, "_coef_imagtime.dat");

    carr_txt(fname, mc->nc, S->C);

    // Record Trap potential

    strcpy(fname, "output/");
    strcat(fname, outfname);
    strcat(fname, "_job1");
    strcat(fname, "_trap.dat");

    rarr_txt(fname,Mdx+1,mc->V);

    fprintf(E_file, "%.10E\n", creal(E0));

    SaveConf(confFileOut, mc);

    // update domain that might have changed
    xi = mc->xi;
    xf = mc->xf;
    dx = mc->dx;

    rarrFillInc(Mdx + 1, xi, dx, x);



    for (i = 1; i < Nlines; i++)
    {

        printf("\n\n\n\n\n\n\n\n\n\n\n\n");
        printf("*****************************      ");
        printf("JOB %d OF %d",i+1,Nlines);
        printf("      *****************************\n\n");

        // number of line reading in _conf.dat and _eq.dat files
        sprintf(strnum,"%d",i+1);

        // release old data
        ReleaseEqDataPkg(mc);

        // setup new parameters
        mc = SetupData(paramFile,confFile,&dt,&N,potname);

        if (Npar != mc->Npar)
        {

            // Setup Coeficients from file because the number of particles
            // changed. The file was left opened,  thus the all vectors of 
            // coefficients for different jobs must be concatenated

            free(S->C);

            // Update number of particles from what was read in input file
            Npar = mc->Npar;
            S->Npar = Npar;

            // Alloc new vector with new configurational size
            S->C = carrDef(NC(Npar, Morb));

            // read concatenated data
            for (k = 0; k < NC(Npar, Morb); k++)
            {
                l = fscanf(coef_file, " (%lf%lfj)", &real, &imag);
                S->C[k] = real + I * imag;
            }
        }
        else
        {

            // The number of particles has not changed though it will
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

        if (resetinit)
        {

            // Read again the same initial orbitals used for all jobs

            strcpy(fname, "input/");
            strcat(fname, infname);
            strcat(fname, "_orb.dat");

            printf("Reseted initial conditions.\n");

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

            // Ortonormalize(Morb,Mdx + 1,dx,S->Omat);
        }
        else
        {
            mc->xi = xi;
            mc->xf = xf;
            mc->dx = dx;

            GetPotential(Mdx+1,mc->Vname,x,mc->V,mc->p[0],mc->p[1],mc->p[2]);
        }



        printf("Configuration successfully set up\n");
        printf("=================================");
        printf("\n# of Particles: %4d",Npar);
        printf("\n# of Orbitals:  %4d",Morb);
        printf("\n# of possible configurations: %d",NC(Npar,Morb));
        printf("\n# of grid points: %d",Mdx+1);
        printf("\nDomain boundaries: [%.2lf,%.2lf]",xi,xf);
        printf("\nGrid step: %.2lf",dx);
        printf("\nFinal time : %.1lf in steps of %.6lf",N*dt,dt);

        // Orthonormality check
        orthoCheck(Npar,Morb,Mdx+1,dx,S->Omat,S->C);

        // Diagonalization with initial orbitals to improve coefficients
        initDiag(mc,S);



        // Call imaginary integrator
        printf("\nStart imaginary time Integration");

        switch (method)
        {

            case 1:

                start = omp_get_wtime();
                s = imagCNSM(mc, S, dt, N, coefInteg, cyclic);
                end = (double) (omp_get_wtime() - start);
                time_used += end;
                break;

            case 2:

                start = omp_get_wtime();
                s = imagCNLU(mc, S, dt, N, coefInteg, cyclic);
                end = (double) (omp_get_wtime() - start);
                time_used += end;
                break;

            case 3:

                start = omp_get_wtime();
                s = imagFFT(mc, S, dt, N, coefInteg);
                end = (double) (omp_get_wtime() - start);
                time_used += end;
                break;
        }

        printf("Time taken in job%d : %.1lf(s) = ",i+1,end);
        TimePrint(end);
        printf("\nAverage per time steps : %.1lf(ms)",end/s*1000);



        // Record data

        E0 = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint);

        // setup filename to store solution
        strcpy(fname, "output/");
        strcat(fname, outfname);
        strcat(fname, "_job");
        strcat(fname, strnum);

        strcat(fname, "_orb_imagtime.dat");
        cmat_txt_T(fname, Morb, Mdx + 1, S->Omat);

        // Record Coeficients Data

        strcpy(fname, "output/");
        strcat(fname, outfname);
        strcat(fname, "_job");
        strcat(fname, strnum);
        strcat(fname, "_coef_imagtime.dat");

        carr_txt(fname, mc->nc, S->C);

        // Record trap potential

        strcpy(fname, "output/");
        strcat(fname, outfname);
        strcat(fname, "_job");
        strcat(fname, strnum);
        strcat(fname, "_trap.dat");

        rarr_txt(fname, Mdx + 1, mc->V);

        fprintf(E_file, "%.10E\n", creal(E0));

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
    free(x);

    printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    printf("***************************      ");
    printf("ALL JOBS DONE");
    printf("      ***************************\n\n");
    printf("\nTotal time taken: %.1lf(min) = ",time_used/60.0);
    TimePrint(time_used);
    
    printf("\n\nAverage time per state: %.1lf(s) = ",time_used/Nlines);
    TimePrint(time_used/Nlines);

    printf("\n\n");
    return 0;
}

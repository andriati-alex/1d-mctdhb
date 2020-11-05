#include "structureSetup.h"



EqDataPkg PackEqData(int Npar, int Morb, int Mpos, double xi, double xf,
          double a2, double g, doublec a1, char Vname[], double p[])
{

/** Create and setup all fields of the Structure with all relevant
    parameters of a configurational problem                    **/

    double
        dx;
    Rarray
        x;

    EqDataPkg MC = (EqDataPkg) malloc(sizeof(struct _EquationDataPkg));

    if (MC == NULL)
    {
        printf("\n\n\nMEMORY ERROR : malloc fail for EqData structure\n\n");
        exit(EXIT_FAILURE);
    }

    dx = (xf - xi) / (Mpos - 1);
    x = rarrDef(Mpos);
    rarrFillInc(Mpos,xi,dx,x);

    // variables transcription
    MC->Npar = Npar;
    MC->Morb = Morb;
    MC->Mpos = Mpos;
    MC->xi = xi;
    MC->xf = xf;
    MC->dx = dx;
    MC->g = g;
    MC->a2 = a2;
    MC->a1 = a1;

    // configurational space
    MC->nc = NC(Npar,Morb);
    MC->NCmat = setupNCmat(Npar,Morb);
    MC->IF = setupFocks(Npar,Morb);
    MC->strideOT = iarrDef(MC->nc);
    MC->strideTT = iarrDef(MC->nc);
    MC->Map = OneOneMap(Npar,Morb,MC->NCmat,MC->IF);
    MC->MapOT = OneTwoMap(Npar,Morb,MC->NCmat,MC->IF,MC->strideOT);
    MC->MapTT = TwoTwoMap(Npar,Morb,MC->NCmat,MC->IF,MC->strideTT);

    // One-body potential
    MC->p[0] = p[0];
    MC->p[1] = p[1];
    MC->p[2] = p[2];
    strcpy(MC->Vname,Vname);
    MC->V = rarrDef(Mpos);
    GetPotential(Mpos,Vname,x,MC->V,p[0],p[1],p[2]);

    free(x);

    return MC;
}



ManyBodyPkg AllocManyBodyPkg(int Npar, int Morb, int Mpos)
{

/** The many-body state expressed in the configurational basis **/

    ManyBodyPkg S = (ManyBodyPkg) malloc(sizeof(struct _ManyBodyDataPkg));

    if (S == NULL)
    {
        printf("\n\n\nMEMORY ERROR : malloc fail ManyBody structure\n\n");
        exit(EXIT_FAILURE);
    }

    S->Npar = Npar;
    S->Morb = Morb;
    S->Mpos = Mpos;
    S->nc   = NC(Npar,Morb);
    S->C    = carrDef(NC(Npar,Morb));
    // orbitals in grid points organized in a matrix
    S->Omat = cmatDef(Morb,Mpos);
    // density matrices
    S->rho1 = cmatDef(Morb,Morb);
    S->rho2 = carrDef(Morb*Morb*Morb*Morb);
    // one- and two-body matrices of orbitals
    S->Ho   = cmatDef(Morb,Morb);
    S->Hint = carrDef(Morb*Morb*Morb*Morb);

    return S;
}

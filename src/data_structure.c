#include "data_structure.h"



EqDataPkg PackEqData(int Npar,int Morb,int Mpos,double xi,double xf,
          double a2,double inter,doublec a1,char Vname[],double p[])
{

/** Return pointer to a basic data structure with all needed information
  * to solve the MCTDHB variational equations.
**/

    Rarray x = rarrDef(Mpos);

    rarrFillInc(Mpos, xi, (xf - xi) / (Mpos - 1), x);

    EqDataPkg MC = (EqDataPkg) malloc(sizeof(struct _EquationDataPkg));

    if (MC == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for EqData structure\n\n");
        exit(EXIT_FAILURE);
    }

    MC->Npar = Npar;
    MC->Morb = Morb;
    MC->Mpos = Mpos;
    MC->xi = xi;
    MC->xf = xf;
    MC->dx = (xf - xi) / (Mpos - 1);
    MC->inter = inter;
    MC->a2 = a2;
    MC->a1 = a1;
    MC->nc = NC(Npar,Morb);
    MC->NCmat = setupNCmat(Npar, Morb);
    MC->IF = setupFocks(Npar, Morb);
    MC->strideOT = iarrDef(MC->nc);
    MC->strideTT = iarrDef(MC->nc);
    MC->Map = OneOneMap(Npar, Morb, MC->NCmat, MC->IF);
    MC->MapOT = OneTwoMap(Npar, Morb, MC->NCmat, MC->IF, MC->strideOT);
    MC->MapTT = TwoTwoMap(Npar, Morb, MC->NCmat, MC->IF, MC->strideTT);

    MC->p[0] = p[0];
    MC->p[1] = p[1];
    MC->p[2] = p[2];
    strcpy(MC->Vname,Vname);

    MC->V = rarrDef(Mpos);

    GetPotential(Mpos, Vname, x, MC->V, p[0], p[1], p[2]);

    free(x);

    return MC;
}





void dpkgEqData(EqDataPkg MC, double * a2, doublec * a1, Rarray V, double * g)
{
    * a2 = MC->a2;
    * a1 = MC->a1;
    * g  = MC->inter;
    V = MC->V;
}





ManyBodyPkg AllocManyBodyPkg(int Npar,int Morb,int Mpos)
{

/** A master structure with all information that defines solution
  * to the MCTDHB equations
**/

    ManyBodyPkg S = (ManyBodyPkg) malloc(sizeof(struct _ManyBodyDataPkg));

    if (S == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail ManyBody structure\n\n");
        exit(EXIT_FAILURE);
    }

    S->Npar = Npar;
    S->Morb = Morb;
    S->Mpos = Mpos;
    S->nc   = NC(Npar,Morb);
    S->C    = carrDef(NC(Npar,Morb));
    S->Omat = cmatDef(Morb,Mpos);
    S->rho1 = cmatDef(Morb,Morb);
    S->rho2 = carrDef(Morb * Morb * Morb * Morb);
    S->Ho   = cmatDef(Morb,Morb);
    S->Hint = carrDef(Morb * Morb * Morb * Morb);

    return S;
}





void ReleaseManyBodyDataPkg (ManyBodyPkg S)
{
    cmatFree(S->Morb,S->Ho);
    cmatFree(S->Morb,S->rho1);
    cmatFree(S->Morb,S->Omat);
    free(S->C);
    free(S->rho2);
    free(S->Hint);
    free(S);
}





void ReleaseEqDataPkg (EqDataPkg MC)
{

    free(MC->IF);

    free(MC->NCmat);

    free(MC->Map);
    free(MC->MapOT);
    free(MC->MapTT);

    free(MC->strideOT);
    free(MC->strideTT);

    free(MC->V);
    free(MC);
}

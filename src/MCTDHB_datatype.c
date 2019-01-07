#include "../include/MCTDHB_datatype.h"



MCTDHBsetup AllocMCTDHBdata ( int Npar,int Morb,int Mpos,double xi,double xf,
            double a2,double inter,double * V,double complex a1 )
{

/** Return pointer to a basic data structure with all needed information
  * to solve the MCTDHB variational equations.
**/

    MCTDHBsetup MC = (MCTDHBsetup) malloc(sizeof(struct _MCTDHBsetup));
    MC->Npar = Npar;
    MC->Morb = Morb;
    MC->Mpos = Mpos;
    MC->xi = xi;
    MC->xf = xf;
    MC->dx = (xf - xi) / (Mpos - 1);
    MC->inter = inter;
    MC->a2 = a2;
    MC->a1 = a1;
    MC->V = V;
    MC->nc = NC(Npar, Morb);
    MC->NCmat = MountNCmat(Npar, Morb);
    MC->IF = MountFocks(Npar, Morb, MC->NCmat);
    return MC;
}





MCTDHBmaster AllocMCTDHBmaster (int Npar,int Morb,int Mpos)
{

/** A master structure with all orbital/coefficients dependent quantities
  * updated iteratively in time steps.
**/

    MCTDHBmaster S = (MCTDHBmaster) malloc(sizeof(struct _MCTDHBmaster));
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





void EraseMCTDHBmaster (MCTDHBmaster S)
{
    cmatFree(S->Morb,S->Ho);
    cmatFree(S->Morb,S->rho1);
    cmatFree(S->Morb,S->Omat);
    free(S->C);
    free(S->rho2);
    free(S->Hint);
    free(S);
}





void EraseMCTDHBdata (MCTDHBsetup MC)
{
    int i;

    for (i = 0; i < MC->nc; i++) free(MC->IF[i]);
    free(MC->IF);

    for (i = 0; i <= MC->Npar; i++) free(MC->NCmat[i]);
    free(MC->NCmat);

    free(MC->V);
    free(MC);
}

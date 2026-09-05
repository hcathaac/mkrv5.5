*TITLE eps-Constraint Method for Multiobjective Optimization (EPSCM,SEQ=319)
$ontext
The eps-Constraint Method

$offtext

$inlinecom [ ]
$eolcom //
$STitle Example model definitions

sets
     p project /1*2437/
     rg regions /ATT, CMK, EMK, THE, NAG, EPI, STE, PEL, CRE, WGR, WMK, ION, SAG/
     lessdev(rg) /EMK, CMK, THE, EPI, WGR/
     trans(rg) /WMK, CRE, ION, PEL, NAG/

     intv intervention / 1*3/
     sec sectors  /1*8/
     crit criteria /1*3/
     ind  indices /1*9/

*GREEN(p) green set
*/
*$include "c:\gams\green2.prn";
*/
*RED(p) red set
*/
*$include "c:\gams\red2.prn";
*/
*GREEN1(GREEN) green set from first round
*/
*$include "c:\gams\green1.prn";
*/

*GREY2(p) grey set
*/
*$include "c:\gams\grey2.prn";
*/
*set
*NEWGREEN(GREEN) the new green elements;
*NEWGREEN(GREEN)= yes;
*NEWGREEN(GREEN1)= no;

*display NEWGREEN;


table budget(p,rg) data matrix with budget for p-project in rg-region
$include "c:\gams\budget.prn" ;

table score(p,crit) data matrix with score for p-project in crit- criterion
$include "c:\gams\score.prn" ;

parameter sector(p)    sector to which p-project belongs
/
$include "c:\gams\sector.prn" ;
/
;
parameter intervention(p)  intervention to which p-project belongs
/
$include "c:\gams\intervention.prn" ;
/
;

parameter  totbudg(sec)
/
1        33000000
2        40000000
3        75000000
4        47000000
5        76000000
6        18000000
7        34000000
8        87000000
/
;

parameter  totbudgi(intv)
/
1        66000000
2       320000000
3        24000000
/
;
parameter totscore(p);

display budget; //('12', 'STE');
display score; //('15', '3');
display intervention; //('25');
display sector; //('32');

Binary Variables
   X(p)      decision variables indicating if project p is selected if eq to 1
Variables
   ZSCORE(crit)     objective function variables for crit-criterion
   PORTFSCORE       score for the whole portfolio

Equations
   budget_lessdev  budget for less developed areas
   budget_trans    budget for areas in transition
   budget_Attica   budget for Attica
   budget_sterea   budget for Sterea
   budget_SAegean  budget for South Aegean

   budget_eq1(sec)     budget for specific sector
   budget_eq2(intv)    budget for specific intervention

   score_eq(crit)     total score for crit-criterion
   totscore_eq        portfolio score equation
;

budget_lessdev.. sum((p,lessdev), budget(p,lessdev)*X(p))   =l= 221400000;
budget_trans..   sum((p,trans), budget(p,trans)*X(p))       =l=  61500000;
budget_Attica..  sum(p, budget(p,'ATT')*X(p))               =l= 106600000;
budget_sterea..  sum(p, budget(p,'STE')*X(p))               =l=  10250000;
budget_SAegean.. sum(p, budget(p,'SAG')*X(p))               =l=  10250000;

budget_eq1(sec)..    sum(p$(sector(p) eq ord(sec)), sum(rg, budget(p,rg))*X(p)) =l= totbudg(sec);
budget_eq2(intv)..   sum(p$(intervention(p) eq ord(intv)), sum(rg, budget(p,rg))*X(p)) =l= totbudgi(intv);

score_eq(crit)..    sum(p, X(p)*score(p,crit)) =e= ZSCORE(crit);
totscore_eq..      sum(p, X(p)*totscore(p)) =e= PORTFSCORE     ;


model itanew /all/;

loop(p,
         if (intervention(p)=1,
             totscore(p)=0.2*score(p,'1')+0.3*score(p,'2')+0.5*score(p,'3');
             );
         if (intervention(p)=2,
             totscore(p)=0.4*score(p,'1')+0.3*score(p,'2')+0.3*score(p,'3');
             );
         if (intervention(p)=3,
             totscore(p)=0.3*score(p,'1')+0.2*score(p,'2')+0.5*score(p,'3');
             );
      );


*X.fx(GREEN)=1;
*X.fx(RED)=0;
*loop(NEWGREEN, budget(NEWGREEN,rg)=0.925*budget(NEWGREEN,rg));
*loop(GREY2, budget(GREY2,rg)=0.85*budget(GREY2,rg));


solve itanew using MIP maximizing PORTFSCORE;

scalar
elapsed_time elapsed time for payoff and e-constraint
z1, z2, z3  auxiliary parameters for random scores
start start time
finish finish time
iter  counter for iterations
*r auxiliary parameter
MCiter number of Monte Carlo iterations /1000/
totiter  total number of MC iterations
;

parameter
totbudgregion(rg)  total budget of region rg
totbudgsector(sec)  total budget of section sec
totbudgintv(intv)  total budget of region rg
;

option seed=5780;
option optcr=0.0005;

FILE fx /c:\gams\proj_2437_out.txt/ ;
fx.pw=10000;
put fx ;

*FILE fx2 /c:\gams\weights.txt/ ;
*fx.pw=1000;

start=jnow;
put 'Monte Carlo iterations' /;
*$ontext
totiter=0 ;
for(iter=1 to MCiter,
* random generation of project scores from uniform distribution

    loop(p,
         z1= score(p,'1')+0.5*uniformint(-2,2);
         if(z1<0, z1=0); if(z1>5, z1=5);
         z2= score(p,'2')+0.5*uniformint(-2,2);
         if(z2<0, z2=0); if(z2>5, z2=5);
         z3= score(p,'3')+0.5*uniformint(-2,2);
         if(z3<0, z3=0); if(z3>5, z3=5);

         if (intervention(p)=1,
             totscore(p)=0.2*z1+0.3*z2+0.5*z3;
             );
         if (intervention(p)=2,
             totscore(p)=0.4*z1+0.3*z2+0.3*z3;
             );
         if (intervention(p)=3,
             totscore(p)=0.3*z1+0.2*z2+0.5*z3;
             );
*          put fx2 ord(p):5, z1:5:2, z2:5:2, z3:5:2, totscore(p):5:3 /
*         if (z1<3 or z2<3 or z3<3, X.FX(p)=0);
*         if (((intervention(p)=3) and (z1<4)), X.FX(p)=0);
         );


    solve itanew using MIP maximizing PORTFSCORE;
    totiter=totiter+1;
    put iter:5:0;
    put fx PORTFSCORE.L:12:3 ;
    loop(p, put X.L(p):3:0);

    loop(rg, totbudgregion(rg)=sum(p, X.L(p)*budget(p,rg)));
    loop(rg, put fx totbudgregion(rg):10:0);
    put '   ';
    totbudgsector(sec)=sum(p$(sector(p) eq ord(sec)), sum(rg, budget(p,rg))*X.L(p));
    loop(sec, put fx totbudgsector(sec):10:0);
    put '   ';
    totbudgintv(intv)=sum(p$(intervention(p) eq ord(intv)), sum(rg, budget(p,rg))*X.L(p));
    loop(intv, put fx totbudgintv(intv):10:0);
    put /;
    );

finish=jnow;
elapsed_time=(finish-start)*86400;
put fx 'Elapsed time: ',elapsed_time:12:2, ' seconds' / ;
put fx 'Monte Carlo total iterations: ', totiter:8:0 /;
putclose fx ;


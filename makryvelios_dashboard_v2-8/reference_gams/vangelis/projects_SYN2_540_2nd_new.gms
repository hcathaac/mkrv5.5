*TITLE eps-Constraint Method for Multiobjective Optimization (EPSCM,SEQ=319)
$ontext
The eps-Constraint Method

$offtext

$inlinecom [ ]
$eolcom //
$STitle Example model definitions

sets
     p project /1*540/
     rg regions /EP2, ATT, CMK, WMK, STE/
     sec sectors  /1*10/
     crit criteria/1*3/

GREEN(p) green set
/
$include "c:\gams\green1_syn2.txt";
/
RED(p) red set
/
$include "c:\gams\red1_syn2.txt";
/
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
$include "c:\gams\budget_syn2.prn" ;

table score(p,crit) data matrix with score for p-project in crit- criterion
$include "c:\gams\score_syn2.prn" ;

parameter sector(p)    sector to which p-project belongs
/
$include "c:\gams\sector_syn2.prn" ;
/
;

parameter  totbudg(sec)
/

1        11664951
2        11664951
3        11664951
4        8748713
5        11664951
6        8748713
7        8748713
8        11644951
9        8748713
10       14581189
/
;

table w(crit,crit)
        1            2            3
1   0.767        0.100        0.100
2   0.100        0.767        0.100
3   0.133        0.133        0.800
;

parameter totscore(p);

display budget; //('12', 'STE');
display score; //('15', '3');
display sector; //('32');

Binary Variables
   X(p)      decision variables indicating if project p is selected if eq to 1
Variables
   PORTFSCORE       score for the whole portfolio

Equations
   budget_EP2      budget for EPANEK2
   budget_Attica   budget for Attica
   budget_CMK      budget for Central Macedonia
   budget_WMK      budget for Western Macedonia
   budget_sterea   budget for Sterea
   totscore_eq        portfolio score equation
;

budget_EP2..     sum(p, budget(p,'EP2')*X(p))   =l=  33413925;
budget_Attica..  sum(p, budget(p,'ATT')*X(p))   =l=  48047006;
budget_CMK..     sum(p, budget(p,'CMK')*X(p))   =l=  22313335;
budget_WMK..     sum(p, budget(p,'WMK')*X(p))   =l=    647220;
budget_sterea..  sum(p, budget(p,'STE')*X(p))   =l=   3480311;
totscore_eq..    sum(p, X(p)*totscore(p)) =e= PORTFSCORE     ;

model itanew /all/;

X.fx(GREEN)=1;
X.fx(RED)=0;
*loop(NEWGREEN, budget(NEWGREEN,rg)=0.925*budget(NEWGREEN,rg));
*loop(GREY2, budget(GREY2,rg)=0.85*budget(GREY2,rg));


scalar
elapsed_time elapsed time for payoff and e-constraint
start start time
finish finish time
;

parameter
totbudgregion(rg)  total budget of region rg
totbudgsector(sec)  total budget of section sec
;

option optcr=0.0000;

FILE fx /c:\gams\proj_SYN2out.txt/ ;
fx.pw=10000;
put fx ;

*FILE fx2 /c:\gams\weights.txt/ ;
*fx.pw=1000;

start=jnow;
put 'Individual Decision Makers Optimizations' /;
*$ontext
loop(crit,
    totscore(p)=w("1",crit)*score(p,"1")+w("2",crit)*score(p,"2")+w("3",crit)*score(p,"3");
    solve itanew using MIP maximizing PORTFSCORE;
    put ord(crit):5:0;
    put fx PORTFSCORE.L:12:3 ;
    loop(p, put X.L(p):3:0);
    loop(rg, totbudgregion(rg)=sum(p, X.L(p)*budget(p,rg)));
    loop(rg, put fx totbudgregion(rg):10:0);
    put '   ';
    totbudgsector(sec)=sum(p$(sector(p) eq ord(sec)), sum(rg, budget(p,rg))*X.L(p));
    loop(sec, put fx totbudgsector(sec):10:0);
    put /;
    );

finish=jnow;
elapsed_time=(finish-start)*86400;
put fx 'Elapsed time: ',elapsed_time:12:2, ' seconds' / ;
putclose fx ;


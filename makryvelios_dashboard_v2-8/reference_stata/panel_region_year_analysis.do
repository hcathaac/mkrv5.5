*****************************************************
* Region–Year Panel Models for EE1, EE4, EE6–EE9
* Poisson / Negative Binomial with region and year effects
* This file constructs a strict Region–Year panel (r,t)
* by collapsing project-level data and then estimating
* count-data models with μ_r and λ_t.
*****************************************************

clear all
set more off

*----------------------------------------------------
* 1. Import project-level Excel data
*----------------------------------------------------

import excel using "Makryvelios  data for R&D Projects and regional development indicators (1).xlsx", ///
    sheet("Data (all)") cell(A2) firstrow clear

* Basic sanity check
describe Region Project_start_year Project_end_year

*----------------------------------------------------
* 2. Define the panel dimensions: Region r and Year t
*    Here Year is taken as the project end year,
*    following the idea that most outcomes (absorption,
*    patents, spin-offs, SMEs benefited) materialise
*    by or at project completion.
*----------------------------------------------------

rename Region                   region
rename Project_end_year         year

* Drop observations without a well-defined region or year
drop if missing(region) | missing(year)

*----------------------------------------------------
* 3. Construct project-level ingredients for Y_rt and X_rt
*----------------------------------------------------

* Absorption ingredients
rename "Final Public expenditure ( at the end)"     final_pubexp
rename "(% absorption rate / public expenditure)"   absorb_rate_pubexp

* If final_pubexp is not numeric, coerce it
capture confirm numeric variable final_pubexp
if _rc {
    destring final_pubexp, replace force
}

* Project-level absorbed amount (proxy):
* absorbed_i ≈ rate_i * final_public_expenditure_i
generate absorbed_amount = absorb_rate_pubexp * final_pubexp

*----------------------------------------------------
* 3.1 Regional characteristics X_rt (will be averaged
*     within Region–Year when collapsing)
*----------------------------------------------------

rename GDP_Region_End_Year                     gdp_end
rename GDP_Region_per_person_End_Year          gdp_pc_end
rename Employment_Region_End_Year              employ_end
rename Researchers_number_Region_End_Year      researchers_end
rename R_D_exp_int_Region_End_Year             rd_intensity_end

* Innovation and collaboration proxies (regional level)
rename Percet_innov_busi_Region_2010_2012      pct_innov_firms
rename Rate_busin_Region_collaborations_with_any_organiz rate_collab_firms
rename Numb_sc_pub_per_Region_2010_2014        scipubs_2010_14
rename Numb_inter_collab_per_Region_2010_2014  intl_collab_2010_14

*----------------------------------------------------
* 3.2 Project-level Controls that will be aggregated
*     to Region–Year (C̄_rt)
*----------------------------------------------------

rename "Final Project's Budjet (at the end of the project)" final_budget
rename "Project Duration  (year)"                            proj_duration

capture confirm numeric variable final_budget
if _rc {
    destring final_budget, replace force
}

* Binary flag: projects involving firms (for C̄_rt)
* We use the number of beneficiary firms as an indicator
* and convert it to a 0–1 flag at project level.
rename Indicator_3106_Nub_comp_benef           num_firm_benef
generate has_firm = (num_firm_benef > 0) if !missing(num_firm_benef)

*----------------------------------------------------
* 3.3 Project-level outcome indicators that will be
*     aggregated to Region–Year outcomes Y_rt
*----------------------------------------------------

* EE4 – number of R&D projects
* (this will be the simple count of projects per Region–Year)

* EE6 – number of collaborative projects
rename Indicator_5_Nub_coop_comp_research_instit  coop_count
generate is_collab_project = (coop_count > 0) if !missing(coop_count)

* EE7 – patents
rename Indicator_3115_Nub_of_patent               num_patents

* EE8 – spin-offs / spin-outs
rename Indicator_3111_Nub_spin_off_spin_outs      num_spinoffs

* EE9 – SMEs benefiting
rename Indicator_3110_Num_of_SMES_benef           num_smes_benef

*----------------------------------------------------
* 4. Collapse to Region–Year panel
*----------------------------------------------------

* For Y_rt:
*   EE1: regional absorption index of public expenditure
*        We approximate this as:
*           Absorption_rt = (Σ_i absorbed_amount_i) / (Σ_i final_pubexp_i)
*   EE4: number of R&D projects = N projects per Region–Year
*   EE6: number of collaborative projects = Σ_i 1{is_collab_project_i}
*   EE7: number of patent applications   = Σ_i num_patents_i
*   EE8: number of spin-offs/spin-outs   = Σ_i num_spinoffs_i
*   EE9: number of SMEs benefiting       = Σ_i num_smes_benef_i
*
* For X_rt:
*   Regional covariates (gdp_end etc.) are identical for all
*   projects in a given Region–Year, so taking the mean is
*   equivalent to taking any single value.
*
* For C̄_rt (aggregated project controls):
*   - avg_budget_rt      = mean(final_budget)
*   - avg_duration_rt    = mean(proj_duration)
*   - share_firmproj_rt  = mean(has_firm)
*----------------------------------------------------

collapse ///
    (sum) absorbed_amount final_pubexp /// EE1 ingredients
    (count) project_count = A.A._Project /// EE4
    (sum) collab_projects = is_collab_project /// EE6
    (sum) patents_rt      = num_patents /// EE7
    (sum) spinoffs_rt     = num_spinoffs /// EE8
    (sum) smes_benef_rt   = num_smes_benef /// EE9
    (mean) gdp_end gdp_pc_end employ_end researchers_end rd_intensity_end /// X_rt economic / R&D
           pct_innov_firms rate_collab_firms scipubs_2010_14 intl_collab_2010_14 /// innovation / knowledge
           final_budget proj_duration has_firm, /// project-level controls (means)
    by(region year)

* Regional absorption index (EE1 outcome)
generate absorption_rt = absorbed_amount / final_pubexp
label var absorption_rt "Regional absorption index (public expenditure)"

* Aggregated controls
rename final_budget   avg_budget_rt
rename proj_duration  avg_duration_rt
rename has_firm       share_firmproj_rt

label var project_count     "Number of R&D projects (EE4)"
label var collab_projects   "Number of collaborative projects (EE6)"
label var patents_rt        "Number of patent applications (EE7)"
label var spinoffs_rt       "Number of spin-offs / spin-outs (EE8)"
label var smes_benef_rt     "Number of SMEs benefiting (EE9)"
label var avg_budget_rt     "Average project budget (Region–Year)"
label var avg_duration_rt   "Average project duration (Region–Year)"
label var share_firmproj_rt "Share of projects with participating firms"

* Drop Region–Years with missing denominators for EE1
drop if missing(absorption_rt)

*----------------------------------------------------
* 5. Declare panel structure
*----------------------------------------------------

encode region, gen(region_id)
xtset region_id year

*----------------------------------------------------
* 6. Define common regressors for panel models
*    X_rt: economic development + R&D capacity + innovation
*    C̄_rt: aggregated project-level controls
*----------------------------------------------------

global X_rt gdp_end gdp_pc_end employ_end researchers_end rd_intensity_end ///
           pct_innov_firms rate_collab_firms scipubs_2010_14 intl_collab_2010_14

global Cbar_rt avg_budget_rt avg_duration_rt share_firmproj_rt

*----------------------------------------------------
* 7. Poisson Region–Year models with μ_r and λ_t
*    Implemented via region and year dummies:
*       i.region_id captures μ_r
*       i.year      captures λ_t
*    Standard errors clustered at Region level.
*----------------------------------------------------

* EE1 – Absorption of public expenditure (non-negative, continuous in (0,1))
*       For illustration, we estimate a Poisson model on a
*       scaled outcome (absorption_rt * 100). In practice,
*       a fractional response model could also be considered,
*       but Poisson with robust SEs is used here for
*       comparability with the count outcomes.
generate absorption_scaled = absorption_rt * 100
label var absorption_scaled "Regional absorption index * 100"

poisson absorption_scaled $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE1_poisson

* EE4 – Number of R&D projects
poisson project_count $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE4_poisson

* EE6 – Collaborative projects
poisson collab_projects $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE6_poisson

* EE7 – Patents
poisson patents_rt $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE7_poisson

* EE8 – Spin-offs / spin-outs
poisson spinoffs_rt $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE8_poisson

* EE9 – SMEs benefiting
poisson smes_benef_rt $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE9_poisson

*----------------------------------------------------
* 8. Negative Binomial models as robustness checks
*    Same specification, but allowing for overdispersion.
*    Again we use region and year effects as dummies and
*    cluster SEs at Region level.
*----------------------------------------------------

nbreg project_count   $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE4_nb

nbreg collab_projects $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE6_nb

nbreg patents_rt      $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE7_nb

nbreg spinoffs_rt     $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE8_nb

nbreg smes_benef_rt   $X_rt $Cbar_rt i.region_id i.year, vce(cluster region_id)
estimates store EE9_nb

* Note: EE1 is not re-estimated with NB because the outcome is
*       a scaled index, not a pure count. The Poisson model
*       with robust SEs is used as a quasi-likelihood estimator.

*----------------------------------------------------
* 9. (Optional) Export Poisson coefficient tables to Excel
*    – structure mirrors previous country-level file.
*----------------------------------------------------

putexcel set "panel_poisson_results.xlsx", replace
putexcel A1 = "Model" B1 = "Variable" C1 = "Coef" D1 = "StdErr" E1 = "z" F1 = "p"

local row = 2
foreach m in EE1_poisson EE4_poisson EE6_poisson EE7_poisson EE8_poisson EE9_poisson {
    estimates restore `m'
    matrix b = e(b)'
    matrix V = e(V)
    local k = rowsof(b)

    forvalues i = 1/`k' {
        local varname : rowname b[`i',1]
        scalar coef = b[`i',1]
        scalar se   = sqrt(V[`i',`i'])
        scalar zval = coef / se
        scalar pval = 2*normal(-abs(zval))

        putexcel A`row' = "`m'"   ///
                B`row' = "`varname'" ///
                C`row' = coef      ///
                D`row' = se        ///
                E`row' = zval      ///
                F`row' = pval
        local row = `row' + 1
    }
}

* End of Region–Year panel analysis .do file
*****************************************************

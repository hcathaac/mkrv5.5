*****************************************************
* Country-wide R&D Projects and Regional Development
* Stata .do file – project-level models (EE1, EE2, EE3, EE5)
* All regions, all projects. All comments in English.
* This mirrors the Attica design but runs on the full sample.
*****************************************************

clear all
set more off

*----------------------------------------------------
* 1. Import Excel data
*    The Excel sheet has a first row with column indices (1–83)
*    and a second row with the actual variable names.
*    The "cell(A2) firstrow" option skips the index row and
*    uses the second row as variable names.
*----------------------------------------------------

import excel using "Makryvelios  data for R&D Projects and regional development indicators (1).xlsx", ///
    sheet("Data (all)") cell(A2) firstrow clear

*----------------------------------------------------
* 2. NO regional restriction here:
*    the analysis is run on all regions (whole country).
*----------------------------------------------------

*----------------------------------------------------
* 3. Clean and harmonise variable names used in the models
*    (only the subset needed for EE1, EE2, EE3, EE5)
*----------------------------------------------------

rename "(% absorption rate / budget)"          absorb_rate_budget
rename Indicator_6_Nub_of_resear_job_IPA       new_research_jobs_IPA
rename Indicator_501_jobs_dur_the_operat_IPA   total_research_employment_IPA
rename Indicator_3106_Nub_comp_benef           num_firm_benef
rename GDP_Region_End_Year                     gdp_end
rename GDP_Region_per_person_End_Year          gdp_pc_end
rename Employment_Region_End_Year              employ_end
rename Researchers_number_Region_End_Year      researchers_end
rename R_D_exp_int_Region_End_Year             rd_intensity_end
rename "Project Duration  (year)"              proj_duration
rename "Final Project's Budjet (at the end of the project)" final_budget

*----------------------------------------------------
* 4. Basic cleaning and construction of controls
*----------------------------------------------------

* Ensure budget is numeric and drop zero / missing values
capture confirm numeric variable final_budget
if _rc {
    destring final_budget, replace force
}

* Drop observations with non-positive budgets (log undefined)
drop if missing(final_budget) | final_budget <= 0

* Log of project budget – used as a scale control
generate ln_budget = ln(final_budget)
label var ln_budget "Log(final project budget)"

* Quick descriptive statistics for documentation
summarize absorb_rate_budget new_research_jobs_IPA total_research_employment_IPA num_firm_benef ///
          gdp_end gdp_pc_end employ_end researchers_end rd_intensity_end proj_duration ln_budget

*----------------------------------------------------
* 5. Common right-hand-side (RHS) specification
*    Regional characteristics + basic project controls
*----------------------------------------------------

global rhs gdp_end gdp_pc_end employ_end researchers_end rd_intensity_end proj_duration ln_budget

*----------------------------------------------------
* 6. Econometric models (OLS with robust SEs)
*    These correspond to the first set of research
*    questions, estimated at project level for
*    the whole country.
*----------------------------------------------------

* EE1 – Absorption of public expenditure (project-level proxy)
regress absorb_rate_budget $rhs, vce(robust)
estimates store EE1

* EE2 – New research jobs (IPA)
regress new_research_jobs_IPA $rhs, vce(robust)
estimates store EE2

* EE3 – Total research employment (IPA during operation)
regress total_research_employment_IPA $rhs, vce(robust)
estimates store EE3

* EE5 – Number of participating firms (beneficiary companies)
regress num_firm_benef $rhs, vce(robust)
estimates store EE5

*----------------------------------------------------
* 7. Export coefficient tables to Excel (built-in putexcel)
*    This produces an .xlsx file with one stacked table
*    containing coefficients, robust standard errors, t-stats
*    and p-values for all four models.
*----------------------------------------------------

putexcel set "country_stata_results.xlsx", replace
putexcel A1 = "Model" B1 = "Variable" C1 = "Coef" D1 = "StdErr" E1 = "t" F1 = "p"

local row = 2
foreach m in EE1 EE2 EE3 EE5 {
    estimates restore `m'
    matrix b = e(b)'
    matrix V = e(V)
    local k = rowsof(b)

    forvalues i = 1/`k' {
        local varname : rowname b[`i',1]
        scalar coef = b[`i',1]
        scalar se   = sqrt(V[`i',`i'])
        scalar tval = coef / se
        scalar pval = 2*ttail(e(df_r), abs(tval))

        putexcel A`row' = "`m'"   ///
                B`row' = "`varname'" ///
                C`row' = coef      ///
                D`row' = se        ///
                E`row' = tval      ///
                F`row' = pval
        local row = `row' + 1
    }
}

* End of country-wide analysis .do file
*****************************************************

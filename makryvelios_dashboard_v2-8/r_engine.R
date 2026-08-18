#!/usr/bin/env Rscript
# Optional R replication engine. The Streamlit app remains fully functional
# without R; when Rscript and the requested packages exist, this produces an
# independent coefficient/fit table for reproducibility checks.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript r_engine.R data.csv config.json output_dir")
data_path <- args[[1]]
config_path <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install R package 'jsonlite'.")
config <- jsonlite::fromJSON(config_path)
d <- read.csv(data_path, check.names = FALSE, stringsAsFactors = FALSE)

quote_name <- function(x) paste0("`", gsub("`", "", x), "`")
rhs <- c(config$x, paste0("factor(", quote_name(config$categorical), ")"))
formula_text <- paste(quote_name(config$y), "~", paste(rhs, collapse = " + "))
f <- as.formula(formula_text)
estimator <- tolower(config$estimator)

if (estimator == "ols") {
  model <- lm(f, data = d)
} else if (estimator == "poisson") {
  model <- glm(f, data = d, family = poisson(link = "log"))
} else if (estimator == "negative binomial") {
  if (!requireNamespace("MASS", quietly = TRUE)) stop("Install R package 'MASS'.")
  model <- MASS::glm.nb(f, data = d)
} else if (estimator %in% c("logit", "fractional logit")) {
  model <- glm(f, data = d, family = binomial(link = "logit"))
} else if (estimator == "probit") {
  model <- glm(f, data = d, family = binomial(link = "probit"))
} else {
  stop(paste("Unsupported R estimator:", config$estimator))
}

coef_table <- as.data.frame(summary(model)$coefficients)
coef_table$term <- rownames(coef_table)
rownames(coef_table) <- NULL
names(coef_table)[1:4] <- c("coefficient", "std_error", "statistic", "p_value")

# HC3 robust covariance when sandwich is available.
if (estimator == "ols" && requireNamespace("sandwich", quietly = TRUE) && requireNamespace("lmtest", quietly = TRUE)) {
  robust <- as.data.frame(lmtest::coeftest(model, vcov. = sandwich::vcovHC(model, type = "HC3")))
  robust$term <- rownames(robust)
  rownames(robust) <- NULL
  names(robust)[1:4] <- c("coefficient", "std_error", "statistic", "p_value")
  coef_table <- robust
}

fit <- data.frame(
  estimator = config$estimator,
  n = stats::nobs(model),
  log_likelihood = as.numeric(logLik(model)),
  AIC = AIC(model),
  BIC = BIC(model),
  r_squared = if (inherits(model, "lm")) summary(model)$r.squared else NA,
  adjusted_r_squared = if (inherits(model, "lm")) summary(model)$adj.r.squared else NA,
  formula = formula_text
)

pred <- data.frame(row_index = as.integer(names(residuals(model))), fitted = fitted(model), residual = residuals(model))
write.csv(coef_table, file.path(output_dir, "r_coefficients.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(fit, file.path(output_dir, "r_fit.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(pred, file.path(output_dir, "r_predictions.csv"), row.names = FALSE, fileEncoding = "UTF-8")
capture.output(summary(model), file = file.path(output_dir, "r_summary.txt"))
writeLines("ok", file.path(output_dir, "SUCCESS"))

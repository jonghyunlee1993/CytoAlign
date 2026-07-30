#!/usr/bin/env Rscript

# Function-level adapter of cyCombine::impute_across_panels.
#
# This deliberately uses only the dependencies required by the panel-merging
# routine (kohonen and base R).  The official package imports the much larger
# Bioconductor batch-correction stack even though that stack is not used here.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 11) {
  stop(
    paste(
      "usage: cycombine_panel_impute.R",
      "REFERENCE_CSV QUERY_CSV OBSERVED_MARKERS HIDDEN_MARKERS OUTPUT_CSV",
      "SEED XDIM YDIM RLEN MIN_REFERENCE_CELLS DISTANCE"
    )
  )
}

reference_path <- args[[1]]
query_path <- args[[2]]
observed_path <- args[[3]]
hidden_path <- args[[4]]
output_path <- args[[5]]
seed <- as.integer(args[[6]])
xdim <- as.integer(args[[7]])
ydim <- as.integer(args[[8]])
rlen <- as.integer(args[[9]])
minimum_reference_cells <- as.integer(args[[10]])
distance <- args[[11]]

read_markers <- function(path) {
  readLines(path, warn = FALSE, encoding = "UTF-8")
}

reference <- read.csv(reference_path, check.names = FALSE)
query <- read.csv(query_path, check.names = FALSE)
observed_markers <- read_markers(observed_path)
hidden_markers <- read_markers(hidden_path)

if (!all(c(observed_markers, hidden_markers) %in% colnames(reference))) {
  stop("Reference data do not contain every requested marker")
}
if (!all(observed_markers %in% colnames(query))) {
  stop("Query data do not contain every observed marker")
}
if (length(hidden_markers) == 0L) {
  stop("At least one hidden marker is required")
}

# Match cyCombine::create_som defaults: kohonen, online mode, 8x8 grid,
# rlen=10, and sum-of-squares distance.
set.seed(seed)
overlap <- rbind(
  reference[, observed_markers, drop = FALSE],
  query[, observed_markers, drop = FALSE]
)
som_model <- kohonen::som(
  as.matrix(overlap),
  grid = kohonen::somgrid(xdim = xdim, ydim = ydim),
  rlen = rlen,
  mode = "online",
  dist.fcts = distance
)
som_classes <- som_model$unit.classif
reference_som <- som_classes[seq_len(nrow(reference))]
query_som <- som_classes[nrow(reference) + seq_len(nrow(query))]

# Match cyCombine::impute_across_panels: within every SOM node, sample a
# complete reference event and add independent Gaussian noise whose bandwidth
# is estimated with stats::density.  The draw is clamped to the node range.
imputed <- matrix(
  NA_real_,
  nrow = nrow(query),
  ncol = length(hidden_markers),
  dimnames = list(NULL, hidden_markers)
)
for (node in sort(unique(query_som))) {
  query_rows <- which(query_som == node)
  reference_rows <- which(reference_som == node)
  if (length(reference_rows) < minimum_reference_cells) {
    next
  }

  bandwidth <- vapply(
    hidden_markers,
    function(marker) stats::density(reference[reference_rows, marker])$bw,
    numeric(1)
  )
  set.seed(seed)
  sampled_rows <- sample(reference_rows, length(query_rows), replace = TRUE)
  draws <- as.matrix(
    reference[sampled_rows, hidden_markers, drop = FALSE]
  )
  noise <- vapply(
    seq_along(hidden_markers),
    function(index) {
      stats::rnorm(length(query_rows), 0, bandwidth[[index]])
    },
    numeric(length(query_rows))
  )
  if (length(hidden_markers) == 1L) {
    noise <- matrix(noise, ncol = 1L)
  }
  draws <- draws + noise
  for (index in seq_along(hidden_markers)) {
    marker <- hidden_markers[[index]]
    lower <- min(reference[reference_rows, marker])
    upper <- max(reference[reference_rows, marker])
    draws[, index] <- pmin(pmax(draws[, index], lower), upper)
  }
  imputed[query_rows, ] <- draws
}

output <- data.frame(
  .row_id = seq_len(nrow(query)) - 1L,
  .som_node = query_som,
  imputed,
  check.names = FALSE
)
write.csv(output, output_path, row.names = FALSE, quote = TRUE)

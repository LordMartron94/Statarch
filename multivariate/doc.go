// Package multivariate provides matrix-based multivariate statistical operations.
//
// This package extends statarch's vector-based operations to handle matrices of features,
// enabling efficient computation of correlation and covariance matrices for datasets with
// multiple variables.
//
// Key operations:
// - Covariance matrices: Compute pairwise covariances between all features
// - Correlation matrices: Compute pairwise correlations between all features
//
// All operations leverage Blaze's matrix multiplication capabilities for optimal performance,
// providing massive speedups over nested loops of pairwise vector operations.
package multivariate



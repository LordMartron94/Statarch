// Package descriptive provides descriptive statistical operations.
//
// All functions operate on StatArchAnalysis structures from statarch/core which cache computed values
// to avoid redundant calculations. Create an analysis structure using core.StatArchAnalysisCreate
// and pass it to the statistical functions.
//
// Functions that work on sorted vectors (median, percentile, IQR) accept the sorted vector directly
// as they may operate on a different vector than the original.
package descriptive

// Package rolling provides rolling statistics (moving windows) for time-series analysis.
//
// All functions operate on StatArchRollingAnalysis structures which cache computed values
// to avoid redundant calculations. Create an analysis structure using StatArchRollingAnalysisCreate
// and pass it to the rolling statistics functions.
//
// Rolling statistics use Welford's online algorithm for numerically stable mean and variance
// computation, maintaining state that updates incrementally as new values are added to the window.
package rolling


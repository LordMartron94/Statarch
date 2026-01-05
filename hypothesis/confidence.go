package hypothesis

import (
	"foundation"
	"math"
	"memcore"
	"statarch/core"
	"statarch/descriptive"
	"sync"

	"gonum.org/v1/gonum/stat/distuv"
)

/*
StatArchHypothesisVectorConfidenceIntervalF64 computes a confidence interval for the mean in float64 precision.

The confidence interval provides a range of values that likely contains the true population mean
with the specified confidence level. It uses the t-distribution for small samples and normal
approximation for large samples.

Use cases:
- Statistical inference (estimating population parameters)
- Performance benchmarking (reporting measurement uncertainty)
- Quality control (process capability analysis)
- Scientific reporting (error bars in graphs)

Time complexity: O(1) - uses cached mean and standard error from analysis
Space complexity: O(1) - only local variables used

Prerequisites:
- analysis must be a valid StatArchAnalysis structure from statarch/core
- Vector must contain at least 2 elements for meaningful interval
- confidence should be in range [0.0, 1.0] (typical values: 0.90, 0.95, 0.99)
- If sample=true, vector represents a sample from a larger population

Edge cases:
- Returns [mean, mean] for empty or single-element vectors
- Returns [mean, mean] if standard error is 0 (no variation)
- Clamps confidence to [0.0, 1.0] range
- Uses t-distribution for n < 30, z-score for n >= 30

Formula: CI = mean ± t_critical * SEM
where t_critical depends on sample size and confidence level
*/
func StatArchHypothesisVectorConfidenceIntervalF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	confidence float64,
	sample bool,
) (lower, upper float64) {
	core.StatArchAnalysisValidateVersion(analysis)

	if analysis.Count < 2 {
		mean := descriptive.StatArchDescriptiveVectorMeanF64(analysis)
		return mean, mean
	}

	// Clamp confidence to valid range
	if confidence < 0.0 {
		confidence = 0.0
	}
	if confidence > 1.0 {
		confidence = 1.0
	}

	mean := descriptive.StatArchDescriptiveVectorMeanF64(analysis)
	sem := descriptive.StatArchDescriptiveVectorStandardErrorMeanF64(analysis, sample)

	if sem == 0.0 {
		return mean, mean
	}

	// Calculate critical value based on sample size and confidence level
	n := float64(analysis.Count)
	criticalValue := computeTCritical(n, confidence)

	margin := criticalValue * sem
	lower = mean - margin
	upper = mean + margin

	return lower, upper
}

/*
StatArchHypothesisVectorIsSignificantlyDifferentF64 determines if two samples are significantly different
using the Mann-Whitney U test (non-parametric).

This function uses the Mann-Whitney U test to determine if two independent samples come from
different distributions. It is a non-parametric alternative to the t-test and does not require
normality assumptions.

Use cases:
- Performance comparison (comparing benchmark results)
- A/B testing (determining if changes are significant)
- Quality control (comparing process outputs)
- Statistical hypothesis testing (when normality cannot be assumed)

Time complexity: O(n log n) - dominated by ranking operation in Mann-Whitney U test
Space complexity: O(n) - requires temporary storage for ranking

Prerequisites:
- Both analysis structures must be valid
- Both vectors must contain at least 1 element
- confidence should be in range [0.0, 1.0]
- allocFn must be provided for creating temporary vectors

Edge cases:
- Returns false if either vector has < 1 element
- Returns false for small samples (n_A + n_B <= 20) where exact distribution would be needed
- Uses p-value threshold based on confidence level: p < (1 - confidence)

Algorithm:
1. Perform Mann-Whitney U test on both samples
2. Extract p-value from test result
3. Compare p-value to significance threshold: (1 - confidence)
4. Return true if p-value < threshold (samples are significantly different)
*/
func StatArchHypothesisVectorIsSignificantlyDifferentF64[T foundation.Numeric](
	analysis1 *core.StatArchAnalysis[T],
	analysis2 *core.StatArchAnalysis[T],
	confidence float64,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) bool {
	core.StatArchAnalysisValidateVersion(analysis1)
	core.StatArchAnalysisValidateVersion(analysis2)

	if analysis1.Count < 1 || analysis2.Count < 1 {
		return false
	}

	// Clamp confidence to valid range
	if confidence < 0.0 {
		confidence = 0.0
	}
	if confidence > 1.0 {
		confidence = 1.0
	}

	// Perform Mann-Whitney U test
	result := StatArchHypothesisVectorMannWhitneyUF64(analysis1, analysis2, allocFn)

	// Check if p-value is valid (not NaN)
	if math.IsNaN(result.PValue) {
		// For small samples, fall back to confidence interval overlap test
		return isSignificantlyDifferentByConfidenceIntervals(analysis1, analysis2, confidence)
	}

	// Significance threshold: p < (1 - confidence)
	// For 95% confidence, we want p < 0.05
	significanceThreshold := 1.0 - confidence
	return result.PValue < significanceThreshold
}

/*
isSignificantlyDifferentByConfidenceIntervals is a fallback method that uses confidence interval overlap
when Mann-Whitney U test cannot provide a p-value (e.g., for small samples).

Time complexity: O(1) - uses cached statistics
Space complexity: O(1) - only local variables
*/
func isSignificantlyDifferentByConfidenceIntervals[T foundation.Numeric](
	analysis1 *core.StatArchAnalysis[T],
	analysis2 *core.StatArchAnalysis[T],
	confidence float64,
) bool {
	lower1, upper1 := StatArchHypothesisVectorConfidenceIntervalF64(analysis1, confidence, true)
	lower2, upper2 := StatArchHypothesisVectorConfidenceIntervalF64(analysis2, confidence, true)

	// If confidence intervals don't overlap, samples are significantly different
	return upper1 < lower2 || upper2 < lower1
}

// criticalValueCache stores cached critical values (t-critical and z-critical) to avoid repeated computations.
// Uses a map-based cache similar to statarch's cache pattern, but for global distribution parameters.
type criticalValueCache struct {
	tValues map[uint64]float64 // key: (df << 16) | (confidence * 1000) as uint64
	zValues map[uint64]float64 // key: confidence * 1000 as uint64
	mu      sync.RWMutex
}

var globalCriticalCache = &criticalValueCache{
	tValues: make(map[uint64]float64),
	zValues: make(map[uint64]float64),
}

// makeTCacheKey creates a cache key for t-critical values from df and confidence.
func makeTCacheKey(df, confidence float64) uint64 {
	// Pack df (as uint16, max 65535) and confidence*1000 (as uint16, max 1000) into uint64
	dfInt := uint64(df)
	confInt := uint64(confidence * 1000.0)
	return (dfInt << 16) | confInt
}

// makeZCacheKey creates a cache key for z-critical values from confidence.
func makeZCacheKey(confidence float64) uint64 {
	return uint64(confidence * 1000.0)
}

/*
computeTCritical computes the critical t-value for a given sample size and confidence level.

Uses t-distribution for small samples (n < 30) and z-score (normal distribution) for large samples.
For both t-distribution and z-score values, uses gonum/stat/distuv library to compute accurate critical values.

Time complexity: O(1) for cached values, O(1) computation for uncached
Space complexity: O(1)

The function caches results using the global critical value cache to avoid repeated computations.
*/
func computeTCritical(n float64, confidence float64) float64 {
	// For large samples (n >= 30), use z-score (normal distribution)
	if n >= 30.0 {
		return computeZCritical(confidence)
	}

	// For small samples, use t-distribution
	df := n - 1.0 // degrees of freedom

	// Check cache first
	cacheKey := makeTCacheKey(df, confidence)
	globalCriticalCache.mu.RLock()
	if cached, ok := globalCriticalCache.tValues[cacheKey]; ok {
		globalCriticalCache.mu.RUnlock()
		return cached
	}
	globalCriticalCache.mu.RUnlock()

	// Calculate quantile probability (two-tailed)
	alpha := 1.0 - confidence
	tailProb := alpha / 2.0
	quantile := 1.0 - tailProb // Upper tail quantile

	// Use gonum StudentsT distribution to compute quantile
	// StudentsT distribution with df degrees of freedom
	studentsT := distuv.StudentsT{
		Mu:    0,   // mean
		Sigma: 1,   // scale
		Nu:    df,  // degrees of freedom
		Src:   nil, // will use default source
	}

	// Compute quantile (inverse CDF)
	tCritical := studentsT.Quantile(quantile)

	// Cache the result
	globalCriticalCache.mu.Lock()
	globalCriticalCache.tValues[cacheKey] = tCritical
	globalCriticalCache.mu.Unlock()

	return tCritical
}

/*
computeZCritical computes the critical z-value (standard normal) for a given confidence level.

Uses gonum/stat/distuv library to compute accurate normal distribution quantiles.

Time complexity: O(1) for cached values, O(1) computation for uncached
Space complexity: O(1)

The function caches results using the global critical value cache to avoid repeated computations.
*/
func computeZCritical(confidence float64) float64 {
	// Check cache first
	cacheKey := makeZCacheKey(confidence)
	globalCriticalCache.mu.RLock()
	if cached, ok := globalCriticalCache.zValues[cacheKey]; ok {
		globalCriticalCache.mu.RUnlock()
		return cached
	}
	globalCriticalCache.mu.RUnlock()

	// Calculate quantile probability (two-tailed)
	alpha := 1.0 - confidence
	tailProb := alpha / 2.0
	quantile := 1.0 - tailProb // Upper tail quantile

	// Use gonum Normal distribution to compute quantile
	// Standard normal distribution (mean=0, stddev=1)
	normal := distuv.Normal{
		Mu:    0,   // mean
		Sigma: 1,   // standard deviation
		Src:   nil, // will use default source
	}

	// Compute quantile (inverse CDF)
	zCritical := normal.Quantile(quantile)

	// Cache the result
	globalCriticalCache.mu.Lock()
	globalCriticalCache.zValues[cacheKey] = zCritical
	globalCriticalCache.mu.Unlock()

	return zCritical
}

package hypothesis

import (
	"foundation"
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

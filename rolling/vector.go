package rolling

import (
	"foundation"
	"memcore"
	"memstruct"
	"statarch/core"
)

/*
StatArchRollingAnalysis holds a circular buffer and its computed rolling statistics.

Functions operate on this structure, computing and storing values as needed to avoid
redundant calculations. Rolling statistics use Welford's online algorithm for
numerically stable mean and variance computation.

Time complexity: O(1) for accessing cached values, O(1) for updating statistics when adding values
Space complexity: O(k) where k is the number of cached statistics (typically small)

Use cases:
- Real-time anomaly detection
- Signal processing
- Financial time-series analysis
- Process monitoring

The structure caches computed values in a map, allowing for unlimited statistics
as the library grows without being constrained by bit flags.

Cache Invalidation:
The structure automatically detects when the underlying buffer has been modified using version tracking.
Each buffer maintains a version number that increments on every modification. The analysis structure
stores the buffer's version when created (SourceVersion). Before returning any cached value, the
analysis validates that the current buffer version matches SourceVersion. If versions don't match,
the entire cache is invalidated and SourceVersion is updated. This ensures that cached statistics
are never stale, even in long-running processes or streaming data scenarios.

Welford State:
The structure maintains Welford's algorithm state (n, M1, M2, M3, M4) for efficient rolling mean, variance, skewness, and kurtosis.
This state is updated incrementally when values are added to the buffer, avoiding full recomputation.
*/
type StatArchRollingAnalysis[T foundation.Numeric] struct {
	Buffer memcore.MarkRaw

	// Type-specific caches to avoid convT32/convT64 overhead
	// Keys are StatKind constants, values are the computed statistics
	CacheF32   map[core.StatKind]float32
	CacheF64   map[core.StatKind]float64
	CacheOther map[core.StatKind]interface{} // For non-float types (T): RollingMin, RollingMax, RollingSum

	// Welford's algorithm state for rolling mean, variance, skewness, and kurtosis
	// These are maintained incrementally as values are added/removed
	WelfordN  float64 // Current count of elements
	WelfordM1 float64 // Running mean
	WelfordM2 float64 // Sum of squared differences (for variance)
	WelfordM3 float64 // Third central moment accumulator (for skewness)
	WelfordM4 float64 // Fourth central moment accumulator (for kurtosis)

	// Running min/max for efficient updates
	RollingMin  T
	RollingMax  T
	MinMaxValid bool // Whether min/max have been initialized

	// Running sum for efficient updates
	RollingSum T

	// Function to allocate memory for scratch vectors if needed
	AllocFn func(sizeBytes, alignment uint64) memcore.MarkRaw

	// SourceVersion stores the version of the buffer when the analysis was created or last validated.
	// This is used to detect when the underlying buffer has been modified, requiring
	// cache invalidation to prevent returning stale cached statistics.
	SourceVersion uint64
}

/*
StatArchRollingAnalysisCreate creates a new analysis structure for a circular buffer.

Time complexity: O(1) - only initializes structure and gets buffer capacity
Space complexity: O(1) - initializes empty map

Parameters:
- buffer: The circular buffer to analyze (must be valid and initialized)
- allocFn: Function to allocate memory for scratch vectors when needed

Returns:
- A new analysis structure ready for use with rolling statistics functions

The structure is initialized with:
- Buffer reference stored
- Empty cache map (no values cached yet)
- Welford state initialized to zero
- Min/max/sum initialized to zero values
- SourceVersion set to the current buffer version for cache invalidation tracking
*/
func StatArchRollingAnalysisCreate[T foundation.Numeric](
	buffer memcore.MarkRaw,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) *StatArchRollingAnalysis[T] {
	return &StatArchRollingAnalysis[T]{
		Buffer:        buffer,
		CacheF32:      make(map[core.StatKind]float32),
		CacheF64:      make(map[core.StatKind]float64),
		CacheOther:    make(map[core.StatKind]interface{}),
		AllocFn:       allocFn,
		WelfordN:      0,
		WelfordM1:     0,
		WelfordM2:     0,
		WelfordM3:     0,
		WelfordM4:     0,
		MinMaxValid:   false,
		SourceVersion: memstruct.CircularBufferVersionGet[T](buffer),
	}
}

/*
StatArchRollingAnalysisCount returns the number of elements in the rolling window.

This is a helper function to access the count from other packages since the count is computed from the buffer.
*/
func StatArchRollingAnalysisCount[T foundation.Numeric](analysis *StatArchRollingAnalysis[T]) uint64 {
	return memstruct.CircularBufferLength[T](analysis.Buffer)
}

/*
StatArchRollingAnalysisValidateVersion checks if the analysis structure's SourceVersion matches
the current buffer version.

If versions match, returns true and the cache is valid.
If versions don't match, invalidates the cache, updates SourceVersion, and returns false.

Time complexity: O(1) - single version comparison
Space complexity: O(1) - no allocations

Returns:
- true if versions match (cache is valid)
- false if versions don't match (cache was invalidated)
*/
func StatArchRollingAnalysisValidateVersion[T foundation.Numeric](analysis *StatArchRollingAnalysis[T]) bool {
	currentVersion := memstruct.CircularBufferVersionGet[T](analysis.Buffer)

	if analysis.SourceVersion == currentVersion {
		return true
	}

	// Version mismatch - invalidate cache and recompute Welford state
	StatArchRollingAnalysisInvalidateCache(analysis)
	return false
}

/*
StatArchRollingAnalysisInvalidateCache clears all cached statistics and recomputes Welford state.

This function is called when the underlying buffer has been modified (version mismatch detected).
It clears the cache map, recomputes Welford state from the current buffer contents, and updates SourceVersion
to the current buffer version.

Time complexity: O(n) where n is the number of elements in the buffer (recomputes Welford state)
Space complexity: O(1) - clears existing cache, doesn't allocate new memory
*/
func StatArchRollingAnalysisInvalidateCache[T foundation.Numeric](analysis *StatArchRollingAnalysis[T]) {
	// Clear all cache maps
	for k := range analysis.CacheF32 {
		delete(analysis.CacheF32, k)
	}
	for k := range analysis.CacheF64 {
		delete(analysis.CacheF64, k)
	}
	for k := range analysis.CacheOther {
		delete(analysis.CacheOther, k)
	}

	// Recompute Welford state from current buffer contents
	bufferLength := memstruct.CircularBufferLength[T](analysis.Buffer)
	analysis.WelfordN = 0
	analysis.WelfordM1 = 0
	analysis.WelfordM2 = 0
	analysis.WelfordM3 = 0
	analysis.WelfordM4 = 0

	var sum T
	var zero T
	analysis.RollingSum = zero
	analysis.MinMaxValid = false

	if bufferLength > 0 {
		// Iterate over buffer to recompute state
		memstruct.CircularBufferUnaryReadOnlyExecute(analysis.Buffer, func(item T) {
			val := float64(item)
			n1 := analysis.WelfordN
			analysis.WelfordN = analysis.WelfordN + 1

			delta := val - analysis.WelfordM1
			delta_n := delta / analysis.WelfordN
			delta_n2 := delta_n * delta_n
			term1 := delta * delta_n * n1

			analysis.WelfordM1 += delta_n
			analysis.WelfordM4 += term1*delta_n2*(analysis.WelfordN*analysis.WelfordN-3*analysis.WelfordN+3) + 6*delta_n2*analysis.WelfordM2 - 4*delta_n*analysis.WelfordM3
			analysis.WelfordM3 += term1*delta_n*(analysis.WelfordN-2) - 3*delta_n*analysis.WelfordM2
			analysis.WelfordM2 += term1

			// Update sum
			sum += item

			// Update min/max
			if !analysis.MinMaxValid {
				analysis.RollingMin = item
				analysis.RollingMax = item
				analysis.MinMaxValid = true
			} else {
				if item < analysis.RollingMin {
					analysis.RollingMin = item
				}
				if item > analysis.RollingMax {
					analysis.RollingMax = item
				}
			}
		}, 1)

		analysis.RollingSum = sum
	}

	// Update SourceVersion to current buffer version
	analysis.SourceVersion = memstruct.CircularBufferVersionGet[T](analysis.Buffer)
}

/*
StatArchRollingVectorMeanF32 computes the rolling mean of a circular buffer in float32 precision using Welford's algorithm.

Use cases:
- Real-time process monitoring
- Signal processing
- Financial time-series analysis
- Anomaly detection

Time complexity: O(1) - uses cached Welford state
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Buffer must contain at least 1 element

Edge cases:
- Returns 0 if buffer is empty
- Works with any numeric type (int, float32, float64, etc.)

The function uses the analysis structure to cache the computed mean value.
If the mean has already been computed and the buffer hasn't changed, it returns the cached value.
*/
func StatArchRollingVectorMeanF32[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
) float32 {
	StatArchRollingAnalysisValidateVersion(analysis)

	if cached, ok := analysis.CacheF32[core.StatKindRollingMeanF32]; ok {
		return cached
	}

	if analysis.WelfordN == 0 {
		analysis.CacheF32[core.StatKindRollingMeanF32] = float32(0)
		analysis.CacheF64[core.StatKindRollingMeanF64] = float64(0)
		return float32(0)
	}

	mean := float32(analysis.WelfordM1)
	analysis.CacheF32[core.StatKindRollingMeanF32] = mean
	analysis.CacheF64[core.StatKindRollingMeanF64] = float64(analysis.WelfordM1)

	return mean
}

/*
StatArchRollingVectorMeanF64 computes the rolling mean of a circular buffer in float64 precision using Welford's algorithm.

Use cases:
- Real-time process monitoring
- Signal processing
- Financial time-series analysis
- Anomaly detection

Time complexity: O(1) - uses cached Welford state
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Buffer must contain at least 1 element

Edge cases:
- Returns 0 if buffer is empty
- Works with any numeric type (int, float32, float64, etc.)

The function uses the analysis structure to cache the computed mean value.
If the mean has already been computed and the buffer hasn't changed, it returns the cached value.
*/
func StatArchRollingVectorMeanF64[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
) float64 {
	StatArchRollingAnalysisValidateVersion(analysis)

	if cached, ok := analysis.CacheF64[core.StatKindRollingMeanF64]; ok {
		return cached
	}

	if analysis.WelfordN == 0 {
		analysis.CacheF32[core.StatKindRollingMeanF32] = float32(0)
		analysis.CacheF64[core.StatKindRollingMeanF64] = float64(0)
		return float64(0)
	}

	mean := analysis.WelfordM1
	analysis.CacheF32[core.StatKindRollingMeanF32] = float32(mean)
	analysis.CacheF64[core.StatKindRollingMeanF64] = mean

	return mean
}

/*
StatArchRollingVectorVarianceF32 computes the rolling variance of a circular buffer in float32 precision using Welford's algorithm.

Use cases:
- Real-time volatility measurement
- Signal processing
- Financial risk analysis
- Anomaly detection

Time complexity: O(1) - uses cached Welford state
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Buffer must contain at least 2 elements for sample variance, 1 for population

Edge cases:
- Returns 0 if buffer has < 2 elements (sample) or < 1 element (population)
- Works with any numeric type (int, float32, float64, etc.)

Parameters:
- sample: If true, computes unbiased sample variance (divides by n-1). If false, population variance (divides by n).

The function uses the analysis structure to cache the computed variance value.
Sample and population variances are cached separately for accuracy.
*/
func StatArchRollingVectorVarianceF32[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
	sample bool,
) float32 {
	StatArchRollingAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindRollingVarianceF32Sample
	} else {
		kind = core.StatKindRollingVarianceF32Population
	}

	if cached, ok := analysis.CacheF32[kind]; ok {
		return cached
	}

	if analysis.WelfordN < 2 && sample {
		analysis.CacheF32[kind] = float32(0)
		analysis.CacheF64[core.StatKindRollingVarianceF64Sample] = float64(0)
		return float32(0)
	}

	if analysis.WelfordN < 1 {
		analysis.CacheF32[kind] = float32(0)
		if sample {
			analysis.CacheF64[core.StatKindRollingVarianceF64Sample] = float64(0)
		} else {
			analysis.CacheF64[core.StatKindRollingVarianceF64Population] = float64(0)
		}
		return float32(0)
	}

	var variance float64
	if sample {
		variance = analysis.WelfordM2 / (analysis.WelfordN - 1)
		analysis.CacheF32[core.StatKindRollingVarianceF32Sample] = float32(variance)
		analysis.CacheF64[core.StatKindRollingVarianceF64Sample] = variance
	} else {
		variance = analysis.WelfordM2 / analysis.WelfordN
		analysis.CacheF32[core.StatKindRollingVarianceF32Population] = float32(variance)
		analysis.CacheF64[core.StatKindRollingVarianceF64Population] = variance
	}

	return float32(variance)
}

/*
StatArchRollingVectorVarianceF64 computes the rolling variance of a circular buffer in float64 precision using Welford's algorithm.

Use cases:
- Real-time volatility measurement
- Signal processing
- Financial risk analysis
- Anomaly detection

Time complexity: O(1) - uses cached Welford state
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Buffer must contain at least 2 elements for sample variance, 1 for population

Edge cases:
- Returns 0 if buffer has < 2 elements (sample) or < 1 element (population)
- Works with any numeric type (int, float32, float64, etc.)

Parameters:
- sample: If true, computes unbiased sample variance (divides by n-1). If false, population variance (divides by n).

The function uses the analysis structure to cache the computed variance value.
Sample and population variances are cached separately for accuracy.
*/
func StatArchRollingVectorVarianceF64[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
	sample bool,
) float64 {
	StatArchRollingAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindRollingVarianceF64Sample
	} else {
		kind = core.StatKindRollingVarianceF64Population
	}

	if cached, ok := analysis.CacheF64[kind]; ok {
		return cached
	}

	if analysis.WelfordN < 2 && sample {
		analysis.CacheF64[kind] = float64(0)
		analysis.CacheF32[core.StatKindRollingVarianceF32Sample] = float32(0)
		return float64(0)
	}

	if analysis.WelfordN < 1 {
		analysis.CacheF64[kind] = float64(0)
		if sample {
			analysis.CacheF32[core.StatKindRollingVarianceF32Sample] = float32(0)
		} else {
			analysis.CacheF32[core.StatKindRollingVarianceF32Population] = float32(0)
		}
		return float64(0)
	}

	var variance float64
	if sample {
		variance = analysis.WelfordM2 / (analysis.WelfordN - 1)
		analysis.CacheF32[core.StatKindRollingVarianceF32Sample] = float32(variance)
		analysis.CacheF64[core.StatKindRollingVarianceF64Sample] = variance
	} else {
		variance = analysis.WelfordM2 / analysis.WelfordN
		analysis.CacheF32[core.StatKindRollingVarianceF32Population] = float32(variance)
		analysis.CacheF64[core.StatKindRollingVarianceF64Population] = variance
	}

	return variance
}

/*
StatArchRollingVectorStandardDeviationF32 computes the rolling standard deviation of a circular buffer in float32 precision.

Use cases:
- Real-time volatility measurement
- Signal processing
- Financial risk analysis
- Anomaly detection

Time complexity: O(1) - uses cached variance
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Buffer must contain at least 2 elements for sample stddev, 1 for population

Edge cases:
- Returns 0 if buffer has < 2 elements (sample) or < 1 element (population)
- Works with any numeric type (int, float32, float64, etc.)

Parameters:
- sample: If true, computes sample standard deviation. If false, population standard deviation.

The function uses the cached variance value from the analysis structure.
Sample and population standard deviations are cached separately for accuracy.
*/
func StatArchRollingVectorStandardDeviationF32[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
	sample bool,
) float32 {
	StatArchRollingAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindRollingStddevF32Sample
	} else {
		kind = core.StatKindRollingStddevF32Population
	}

	if cached, ok := analysis.CacheF32[kind]; ok {
		return cached
	}

	variance := StatArchRollingVectorVarianceF32(analysis, sample)
	stddev := float32(foundation.Sqrt64(float64(variance)))

	analysis.CacheF32[kind] = stddev
	if sample {
		analysis.CacheF64[core.StatKindRollingStddevF64Sample] = float64(stddev)
	} else {
		analysis.CacheF64[core.StatKindRollingStddevF64Population] = float64(stddev)
	}

	return stddev
}

/*
StatArchRollingVectorStandardDeviationF64 computes the rolling standard deviation of a circular buffer in float64 precision.

Use cases:
- Real-time volatility measurement
- Signal processing
- Financial risk analysis
- Anomaly detection

Time complexity: O(1) - uses cached variance
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Buffer must contain at least 2 elements for sample stddev, 1 for population

Edge cases:
- Returns 0 if buffer has < 2 elements (sample) or < 1 element (population)
- Works with any numeric type (int, float32, float64, etc.)

Parameters:
- sample: If true, computes sample standard deviation. If false, population standard deviation.

The function uses the cached variance value from the analysis structure.
Sample and population standard deviations are cached separately for accuracy.
*/
func StatArchRollingVectorStandardDeviationF64[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
	sample bool,
) float64 {
	StatArchRollingAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindRollingStddevF64Sample
	} else {
		kind = core.StatKindRollingStddevF64Population
	}

	if cached, ok := analysis.CacheF64[kind]; ok {
		return cached
	}

	variance := StatArchRollingVectorVarianceF64(analysis, sample)
	stddev := foundation.Sqrt64(variance)

	analysis.CacheF64[kind] = stddev
	if sample {
		analysis.CacheF32[core.StatKindRollingStddevF32Sample] = float32(stddev)
	} else {
		analysis.CacheF32[core.StatKindRollingStddevF32Population] = float32(stddev)
	}

	return stddev
}

/*
StatArchRollingVectorMin returns the minimum value in the rolling window.

Use cases:
- Real-time minimum tracking
- Signal processing
- Financial analysis (lowest price, etc.)

Time complexity: O(1) - uses cached min value
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Buffer must contain at least 1 element

Edge cases:
- Returns zero value if buffer is empty
- Works with any numeric type (int, float32, float64, etc.)

The function maintains a running minimum that updates incrementally as values are added/removed.
*/
func StatArchRollingVectorMin[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
) T {
	StatArchRollingAnalysisValidateVersion(analysis)

	if cached, ok := analysis.CacheOther[core.StatKindRollingMin]; ok {
		return cached.(T)
	}

	if !analysis.MinMaxValid {
		var zero T
		analysis.CacheOther[core.StatKindRollingMin] = zero
		return zero
	}

	analysis.CacheOther[core.StatKindRollingMin] = analysis.RollingMin
	return analysis.RollingMin
}

/*
StatArchRollingVectorMax returns the maximum value in the rolling window.

Use cases:
- Real-time maximum tracking
- Signal processing
- Financial analysis (highest price, etc.)

Time complexity: O(1) - uses cached max value
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Buffer must contain at least 1 element

Edge cases:
- Returns zero value if buffer is empty
- Works with any numeric type (int, float32, float64, etc.)

The function maintains a running maximum that updates incrementally as values are added/removed.
*/
func StatArchRollingVectorMax[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
) T {
	StatArchRollingAnalysisValidateVersion(analysis)

	if cached, ok := analysis.CacheOther[core.StatKindRollingMax]; ok {
		return cached.(T)
	}

	if !analysis.MinMaxValid {
		var zero T
		analysis.CacheOther[core.StatKindRollingMax] = zero
		return zero
	}

	analysis.CacheOther[core.StatKindRollingMax] = analysis.RollingMax
	return analysis.RollingMax
}

/*
StatArchRollingVectorSum returns the sum of all values in the rolling window.

Use cases:
- Real-time sum tracking
- Signal processing
- Financial analysis (cumulative values, etc.)

Time complexity: O(1) - uses cached sum value
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Buffer must contain at least 1 element

Edge cases:
- Returns zero value if buffer is empty
- Works with any numeric type (int, float32, float64, etc.)

The function maintains a running sum that updates incrementally as values are added/removed.
*/
func StatArchRollingVectorSum[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
) T {
	StatArchRollingAnalysisValidateVersion(analysis)

	if cached, ok := analysis.CacheOther[core.StatKindRollingSum]; ok {
		return cached.(T)
	}

	analysis.CacheOther[core.StatKindRollingSum] = analysis.RollingSum
	return analysis.RollingSum
}

/*
StatArchRollingVectorSkewnessF32 computes the rolling skewness of a circular buffer in float32 precision.

Skewness measures the asymmetry of the distribution. Positive skewness indicates a longer tail on the right,
negative skewness indicates a longer tail on the left.

Use cases:
- Real-time detection of distribution shape changes
- Anomaly detection (detecting when behavior changes in "nature" not just magnitude)
- Financial analysis (detecting fat tails, black swan events)
- Process monitoring (detecting when system behavior becomes asymmetric)

Time complexity: O(1) - uses cached Welford state
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Buffer must contain at least 3 elements for sample skewness, 2 for population

Edge cases:
- Returns 0 if buffer has < 3 elements (sample) or < 2 elements (population)
- Returns 0 if standard deviation is 0
- Works with any numeric type (int, float32, float64, etc.)

Parameters:
- sample: If true, computes unbiased sample skewness. If false, population skewness.

The function uses the analysis structure to cache the computed skewness value.
Sample and population skewness are cached separately for accuracy.
*/
func StatArchRollingVectorSkewnessF32[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
	sample bool,
) float32 {
	StatArchRollingAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindRollingSkewnessF32Sample
	} else {
		kind = core.StatKindRollingSkewnessF32Population
	}

	if cached, ok := analysis.CacheF32[kind]; ok {
		return cached
	}

	if analysis.WelfordN < 3 && sample {
		analysis.CacheF32[kind] = float32(0)
		analysis.CacheF64[core.StatKindRollingSkewnessF64Sample] = float64(0)
		return float32(0)
	}

	if analysis.WelfordN < 2 {
		analysis.CacheF32[kind] = float32(0)
		if sample {
			analysis.CacheF64[core.StatKindRollingSkewnessF64Sample] = float64(0)
		} else {
			analysis.CacheF64[core.StatKindRollingSkewnessF64Population] = float64(0)
		}
		return float32(0)
	}

	// Check if M2 is zero (all values identical)
	if analysis.WelfordM2 == 0 {
		analysis.CacheF32[kind] = float32(0)
		if sample {
			analysis.CacheF64[core.StatKindRollingSkewnessF64Sample] = float64(0)
		} else {
			analysis.CacheF64[core.StatKindRollingSkewnessF64Population] = float64(0)
		}
		return float32(0)
	}

	var skewness float64
	if sample {
		// Sample skewness: √(n(n-1))/(n-2) * (M3 / M2^(3/2))
		// This matches the formula used in AnalyzeWelford for consistency
		skewness = (foundation.Sqrt64(analysis.WelfordN*(analysis.WelfordN-1)) / (analysis.WelfordN - 2)) * (analysis.WelfordM3 / foundation.Pow64(analysis.WelfordM2, 1.5))
		analysis.CacheF32[core.StatKindRollingSkewnessF32Sample] = float32(skewness)
		analysis.CacheF64[core.StatKindRollingSkewnessF64Sample] = skewness
	} else {
		// Population skewness: M3 * √n / M2^(3/2)
		// Equivalent to: M3 / ((M2/n)^(3/2) * n) = M3 * n^(1/2) / M2^(3/2)
		skewness = analysis.WelfordM3 * foundation.Sqrt64(analysis.WelfordN) / foundation.Pow64(analysis.WelfordM2, 1.5)
		analysis.CacheF32[core.StatKindRollingSkewnessF32Population] = float32(skewness)
		analysis.CacheF64[core.StatKindRollingSkewnessF64Population] = skewness
	}

	return float32(skewness)
}

/*
StatArchRollingVectorSkewnessF64 computes the rolling skewness of a circular buffer in float64 precision.

See StatArchRollingVectorSkewnessF32 for detailed documentation.
This is the float64 variant for higher precision requirements.
*/
func StatArchRollingVectorSkewnessF64[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
	sample bool,
) float64 {
	StatArchRollingAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindRollingSkewnessF64Sample
	} else {
		kind = core.StatKindRollingSkewnessF64Population
	}

	if cached, ok := analysis.CacheF64[kind]; ok {
		return cached
	}

	if analysis.WelfordN < 3 && sample {
		analysis.CacheF64[kind] = float64(0)
		analysis.CacheF32[core.StatKindRollingSkewnessF32Sample] = float32(0)
		return float64(0)
	}

	if analysis.WelfordN < 2 {
		analysis.CacheF64[kind] = float64(0)
		if sample {
			analysis.CacheF32[core.StatKindRollingSkewnessF32Sample] = float32(0)
		} else {
			analysis.CacheF32[core.StatKindRollingSkewnessF32Population] = float32(0)
		}
		return float64(0)
	}

	// Check if M2 is zero (all values identical)
	if analysis.WelfordM2 == 0 {
		analysis.CacheF64[kind] = float64(0)
		if sample {
			analysis.CacheF32[core.StatKindRollingSkewnessF32Sample] = float32(0)
		} else {
			analysis.CacheF32[core.StatKindRollingSkewnessF32Population] = float32(0)
		}
		return float64(0)
	}

	var skewness float64
	if sample {
		// Sample skewness: √(n(n-1))/(n-2) * (M3 / M2^(3/2))
		// This matches the formula used in AnalyzeWelford for consistency
		skewness = (foundation.Sqrt64(analysis.WelfordN*(analysis.WelfordN-1)) / (analysis.WelfordN - 2)) * (analysis.WelfordM3 / foundation.Pow64(analysis.WelfordM2, 1.5))
		analysis.CacheF32[core.StatKindRollingSkewnessF32Sample] = float32(skewness)
		analysis.CacheF64[core.StatKindRollingSkewnessF64Sample] = skewness
	} else {
		// Population skewness: M3 * √n / M2^(3/2)
		// Equivalent to: M3 / ((M2/n)^(3/2) * n) = M3 * n^(1/2) / M2^(3/2)
		skewness = analysis.WelfordM3 * foundation.Sqrt64(analysis.WelfordN) / foundation.Pow64(analysis.WelfordM2, 1.5)
		analysis.CacheF32[core.StatKindRollingSkewnessF32Population] = float32(skewness)
		analysis.CacheF64[core.StatKindRollingSkewnessF64Population] = skewness
	}

	return skewness
}

/*
StatArchRollingVectorKurtosisF32 computes the rolling kurtosis of a circular buffer in float32 precision.

Kurtosis measures the "tailedness" of the distribution. High kurtosis indicates heavy tails (more outliers),
low kurtosis indicates light tails. Excess kurtosis (kurtosis - 3) is often used, where 0 indicates normal distribution.

Use cases:
- Real-time detection of fat tails (black swan events)
- Anomaly detection (detecting when distribution shape changes)
- Financial risk analysis (detecting extreme events)
- Process monitoring (detecting when system behavior becomes more extreme)

Time complexity: O(1) - uses cached Welford state
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Buffer must contain at least 4 elements for sample kurtosis, 3 for population

Edge cases:
- Returns 0 if buffer has < 4 elements (sample) or < 3 elements (population)
- Returns 0 if standard deviation is 0
- Works with any numeric type (int, float32, float64, etc.)

Parameters:
- sample: If true, computes unbiased sample kurtosis. If false, population kurtosis.

The function uses the analysis structure to cache the computed kurtosis value.
Sample and population kurtosis are cached separately for accuracy.
*/
func StatArchRollingVectorKurtosisF32[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
	sample bool,
) float32 {
	StatArchRollingAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindRollingKurtosisF32Sample
	} else {
		kind = core.StatKindRollingKurtosisF32Population
	}

	if cached, ok := analysis.CacheF32[kind]; ok {
		return cached
	}

	if analysis.WelfordN < 4 && sample {
		analysis.CacheF32[kind] = float32(0)
		analysis.CacheF64[core.StatKindRollingKurtosisF64Sample] = float64(0)
		return float32(0)
	}

	if analysis.WelfordN < 3 {
		analysis.CacheF32[kind] = float32(0)
		if sample {
			analysis.CacheF64[core.StatKindRollingKurtosisF64Sample] = float64(0)
		} else {
			analysis.CacheF64[core.StatKindRollingKurtosisF64Population] = float64(0)
		}
		return float32(0)
	}

	// Check if M2 is zero (all values identical)
	if analysis.WelfordM2 == 0 {
		analysis.CacheF32[kind] = float32(0)
		if sample {
			analysis.CacheF64[core.StatKindRollingKurtosisF64Sample] = float64(0)
		} else {
			analysis.CacheF64[core.StatKindRollingKurtosisF64Population] = float64(0)
		}
		return float32(0)
	}

	var kurtosis float64
	if sample {
		// Sample excess kurtosis: [n(n+1)(n-1)/((n-2)(n-3))] * (M4/M2²) - 3(n-1)²/((n-2)(n-3))
		// This matches the formula used in AnalyzeWelford for consistency
		term1 := (analysis.WelfordN * (analysis.WelfordN + 1) * (analysis.WelfordN - 1)) / ((analysis.WelfordN - 2) * (analysis.WelfordN - 3))
		term2 := (3 * (analysis.WelfordN - 1) * (analysis.WelfordN - 1)) / ((analysis.WelfordN - 2) * (analysis.WelfordN - 3))
		kurtosis = (term1 * analysis.WelfordM4 / (analysis.WelfordM2 * analysis.WelfordM2)) - term2
		analysis.CacheF32[core.StatKindRollingKurtosisF32Sample] = float32(kurtosis)
		analysis.CacheF64[core.StatKindRollingKurtosisF64Sample] = kurtosis
	} else {
		// Population excess kurtosis: (n * M4 / M2²) - 3
		// This matches the formula used in AnalyzeWelford for consistency
		kurtosis = (analysis.WelfordN * analysis.WelfordM4 / (analysis.WelfordM2 * analysis.WelfordM2)) - 3
		analysis.CacheF32[core.StatKindRollingKurtosisF32Population] = float32(kurtosis)
		analysis.CacheF64[core.StatKindRollingKurtosisF64Population] = kurtosis
	}

	return float32(kurtosis)
}

/*
StatArchRollingVectorKurtosisF64 computes the rolling kurtosis of a circular buffer in float64 precision.

See StatArchRollingVectorKurtosisF32 for detailed documentation.
This is the float64 variant for higher precision requirements.
*/
func StatArchRollingVectorKurtosisF64[T foundation.Numeric](
	analysis *StatArchRollingAnalysis[T],
	sample bool,
) float64 {
	StatArchRollingAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindRollingKurtosisF64Sample
	} else {
		kind = core.StatKindRollingKurtosisF64Population
	}

	if cached, ok := analysis.CacheF64[kind]; ok {
		return cached
	}

	if analysis.WelfordN < 4 && sample {
		analysis.CacheF64[kind] = float64(0)
		analysis.CacheF32[core.StatKindRollingKurtosisF32Sample] = float32(0)
		return float64(0)
	}

	if analysis.WelfordN < 3 {
		analysis.CacheF64[kind] = float64(0)
		if sample {
			analysis.CacheF32[core.StatKindRollingKurtosisF32Sample] = float32(0)
		} else {
			analysis.CacheF32[core.StatKindRollingKurtosisF32Population] = float32(0)
		}
		return float64(0)
	}

	// Check if M2 is zero (all values identical)
	if analysis.WelfordM2 == 0 {
		analysis.CacheF64[kind] = float64(0)
		if sample {
			analysis.CacheF32[core.StatKindRollingKurtosisF32Sample] = float32(0)
		} else {
			analysis.CacheF32[core.StatKindRollingKurtosisF32Population] = float32(0)
		}
		return float64(0)
	}

	var kurtosis float64
	if sample {
		// Sample excess kurtosis: [n(n+1)(n-1)/((n-2)(n-3))] * (M4/M2²) - 3(n-1)²/((n-2)(n-3))
		// This matches the formula used in AnalyzeWelford for consistency
		term1 := (analysis.WelfordN * (analysis.WelfordN + 1) * (analysis.WelfordN - 1)) / ((analysis.WelfordN - 2) * (analysis.WelfordN - 3))
		term2 := (3 * (analysis.WelfordN - 1) * (analysis.WelfordN - 1)) / ((analysis.WelfordN - 2) * (analysis.WelfordN - 3))
		kurtosis = (term1 * analysis.WelfordM4 / (analysis.WelfordM2 * analysis.WelfordM2)) - term2
		analysis.CacheF32[core.StatKindRollingKurtosisF32Sample] = float32(kurtosis)
		analysis.CacheF64[core.StatKindRollingKurtosisF64Sample] = kurtosis
	} else {
		// Population excess kurtosis: (n * M4 / M2²) - 3
		// This matches the formula used in AnalyzeWelford for consistency
		kurtosis = (analysis.WelfordN * analysis.WelfordM4 / (analysis.WelfordM2 * analysis.WelfordM2)) - 3
		analysis.CacheF32[core.StatKindRollingKurtosisF32Population] = float32(kurtosis)
		analysis.CacheF64[core.StatKindRollingKurtosisF64Population] = kurtosis
	}

	return kurtosis
}

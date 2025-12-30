package descriptive

import (
	"blaze/math"
	"blaze/reduce"
	"foundation"
	"memcore"
	"memstruct"
	"statarch/core"
)

/*
StatArchDescriptiveVectorMeanF32 computes the arithmetic mean of a vector in float32 precision.

Use cases:
- Measuring central tendency of data
- Quality control and process monitoring
- Feature analysis in machine learning
- Financial computing (average returns, prices)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty (count is 0)
- Works with any numeric type (int, float32, float64, etc.)

The function uses the analysis structure to cache the computed mean value.
If the mean has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorMeanF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMeanF32); ok {
		return cached.(float32)
	}

	mean := reduce.BlazeReduceVectorMeanF32[T](analysis.Vector)

	core.SetCacheValue(analysis, core.StatKindMeanF32, mean)

	return mean
}

/*
StatArchDescriptiveVectorMeanF64 computes the arithmetic mean of a vector in float64 precision.

Use cases:
- Measuring central tendency of data
- Quality control and process monitoring
- Feature analysis in machine learning
- Financial computing (average returns, prices)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty (count is 0)
- Works with any numeric type (int, float32, float64, etc.)

The function uses the analysis structure to cache the computed mean value.
If the mean has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorMeanF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMeanF64); ok {
		return cached.(float64)
	}

	mean := reduce.BlazeReduceVectorMeanF64[T](analysis.Vector)

	core.SetCacheValue(analysis, core.StatKindMeanF64, mean)

	return mean
}

/*
StatArchDescriptiveVectorSumF32 computes the sum of all elements in a vector in float32 precision.

Use cases:
- Pre-requisite for distance functions (Bhattacharyya, KL divergence)
- Normalization of probability distributions
- Cumulative value tracking

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty (count is 0)
- Works with any numeric type (int, float32, float64, etc.)

The function uses the analysis structure to cache the computed sum value.
If the sum has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorSumF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindSumF32); ok {
		return cached.(float32)
	}

	sum := reduce.BlazeReduceVectorSumF32[T](analysis.Vector)

	core.SetCacheValue(analysis, core.StatKindSumF32, sum)

	return sum
}

/*
StatArchDescriptiveVectorSumF64 computes the sum of all elements in a vector in float64 precision.

Use cases:
- Pre-requisite for distance functions (Bhattacharyya, KL divergence)
- Normalization of probability distributions
- Cumulative value tracking

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty (count is 0)
- Works with any numeric type (int, float32, float64, etc.)

The function uses the analysis structure to cache the computed sum value.
If the sum has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorSumF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindSumF64); ok {
		return cached.(float64)
	}

	sum := reduce.BlazeReduceVectorSumF64[T](analysis.Vector)

	core.SetCacheValue(analysis, core.StatKindSumF64, sum)

	return sum
}

/*
StatArchDescriptiveVectorNormSquaredF32 computes the sum of squares (norm squared) of a vector in float32 precision.

The norm squared is the sum of each element squared: ||v||² = Σ(v_i²)

Use cases:
- Pre-requisite for cosine similarity (vector norm)
- Distance calculations
- Vector magnitude computations

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty (count is 0)
- Works with any numeric type (int, float32, float64, etc.)

The function uses the analysis structure to cache the computed norm squared value.
If the norm squared has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorNormSquaredF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindNormSquaredF32); ok {
		return cached.(float32)
	}

	normSquared := reduce.BlazeReduceVectorSumSquaredF32[T](analysis.Vector)

	core.SetCacheValue(analysis, core.StatKindNormSquaredF32, normSquared)

	return normSquared
}

/*
StatArchDescriptiveVectorNormSquaredF64 computes the sum of squares (norm squared) of a vector in float64 precision.

The norm squared is the sum of each element squared: ||v||² = Σ(v_i²)

Use cases:
- Pre-requisite for cosine similarity (vector norm)
- Distance calculations
- Vector magnitude computations

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty (count is 0)
- Works with any numeric type (int, float32, float64, etc.)

The function uses the analysis structure to cache the computed norm squared value.
If the norm squared has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorNormSquaredF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindNormSquaredF64); ok {
		return cached.(float64)
	}

	normSquared := reduce.BlazeReduceVectorSumSquaredF64[T](analysis.Vector)

	core.SetCacheValue(analysis, core.StatKindNormSquaredF64, normSquared)

	return normSquared
}

/*
StatArchDescriptiveVectorHarmonicMeanF32 computes the harmonic mean of a vector in float32 precision.

The harmonic mean is the reciprocal of the arithmetic mean of reciprocals.
Formula: n / (1/x1 + 1/x2 + ... + 1/xn)

Use cases:
- Averaging rates (speed, efficiency, ratios)
- Financial analysis (average price-to-earnings ratios)
- Physics (average resistance in parallel circuits)
- Data analysis (when dealing with rates or ratios)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 1 element
- All elements must be non-zero (zero values cause division by zero)
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty
- Panics if any element is zero (division by zero in reciprocal calculation)
- Harmonic mean is always less than or equal to arithmetic mean
- Works best with positive values

The function uses the analysis structure to cache the computed harmonic mean value.
If the harmonic mean has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorHarmonicMeanF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindHarmonicMeanF32); ok {
		return cached.(float32)
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	var sumReciprocals float32 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		val := float32(item)
		if val == 0 {
			panic("harmonic mean cannot be computed with zero values")
		}
		sumReciprocals += 1.0 / val
	}, core.StatarchDefaultStride)

	harmonicMean := float32(size) / sumReciprocals

	core.SetCacheValue(analysis, core.StatKindHarmonicMeanF32, harmonicMean)

	return harmonicMean
}

/*
StatArchDescriptiveVectorHarmonicMeanF64 computes the harmonic mean of a vector in float64 precision.

The harmonic mean is the reciprocal of the arithmetic mean of reciprocals.
Formula: n / (1/x1 + 1/x2 + ... + 1/xn)

Use cases:
- Averaging rates (speed, efficiency, ratios)
- Financial analysis (average price-to-earnings ratios)
- Physics (average resistance in parallel circuits)
- Data analysis (when dealing with rates or ratios)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 1 element
- All elements must be non-zero (zero values cause division by zero)
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty
- Panics if any element is zero (division by zero in reciprocal calculation)
- Harmonic mean is always less than or equal to arithmetic mean
- Works best with positive values

The function uses the analysis structure to cache the computed harmonic mean value.
If the harmonic mean has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorHarmonicMeanF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindHarmonicMeanF64); ok {
		return cached.(float64)
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	var sumReciprocals float64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		val := float64(item)
		if val == 0 {
			panic("harmonic mean cannot be computed with zero values")
		}
		sumReciprocals += 1.0 / val
	}, core.StatarchDefaultStride)

	harmonicMean := float64(size) / sumReciprocals
	core.SetCacheValue(analysis, core.StatKindHarmonicMeanF64, harmonicMean)

	return harmonicMean
}

/*
StatArchDescriptiveVectorGeometricMeanF32 computes the geometric mean of a vector in float32 precision.

The geometric mean is the nth root of the product of n values.
Formula: (x1 * x2 * ... * xn)^(1/n)

Use cases:
- Averaging ratios, percentages, or multiplicative factors
- Financial analysis (compound interest rates, portfolio returns)
- Growth rates (population, economic indicators)
- Data normalization (when values span multiple orders of magnitude)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 1 element
- All elements must be positive (zero or negative values cause undefined results)
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty
- Panics if any element is zero or negative (geometric mean undefined for non-positive values)
- Geometric mean is always less than or equal to arithmetic mean
- For positive values, geometric mean provides a better average for multiplicative processes

The function uses the analysis structure to cache the computed geometric mean value.
If the geometric mean has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorGeometricMeanF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindGeometricMeanF32); ok {
		return cached.(float32)
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	var sumLogs float32 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		val := float32(item)
		if val <= 0 {
			panic("geometric mean cannot be computed with zero or negative values")
		}
		sumLogs += foundation.Log32(val)
	}, core.StatarchDefaultStride)

	geometricMean := foundation.Exp32(sumLogs / float32(size))
	core.SetCacheValue(analysis, core.StatKindGeometricMeanF32, geometricMean)

	return geometricMean
}

/*
StatArchDescriptiveVectorGeometricMeanF64 computes the geometric mean of a vector in float64 precision.

The geometric mean is the nth root of the product of n values.
Formula: (x1 * x2 * ... * xn)^(1/n)

Use cases:
- Averaging ratios, percentages, or multiplicative factors
- Financial analysis (compound interest rates, portfolio returns)
- Growth rates (population, economic indicators)
- Data normalization (when values span multiple orders of magnitude)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 1 element
- All elements must be positive (zero or negative values cause undefined results)
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty
- Panics if any element is zero or negative (geometric mean undefined for non-positive values)
- Geometric mean is always less than or equal to arithmetic mean
- For positive values, geometric mean provides a better average for multiplicative processes

The function uses the analysis structure to cache the computed geometric mean value.
If the geometric mean has already been computed, it returns the cached value.
*/
func StatArchDescriptiveVectorGeometricMeanF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindGeometricMeanF64); ok {
		return cached.(float64)
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	var sumLogs float64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		val := float64(item)
		if val <= 0 {
			panic("geometric mean cannot be computed with zero or negative values")
		}
		sumLogs += foundation.Log64(val)
	}, core.StatarchDefaultStride)

	geometricMean := foundation.Exp64(sumLogs / float64(size))
	core.SetCacheValue(analysis, core.StatKindGeometricMeanF64, geometricMean)

	return geometricMean
}

/*
StatArchDescriptiveVectorTrimmedMeanF32 computes the trimmed mean (mean after removing outliers) in float32 precision.

The trimmed mean removes a specified percentage of the smallest and largest values before calculating
the mean, making it more robust to outliers than the standard mean.

Use cases:
- Robust measure of central tendency (less sensitive to outliers than regular mean)
- Quality control (removing measurement errors)
- Financial analysis (removing extreme market events)
- Data preprocessing (outlier-resistant feature engineering)

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- trimPercent must be in range [0.0, 0.5] (percentage to trim from each end)
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if trimPercent < 0 or trimPercent > 0.5
- If trimPercent = 0, returns the same as regular mean
- If trimPercent = 0.5, returns the median (if vector has even length, returns mean of two middle values)
- If trimming would remove all elements, returns 0

Note: Trimmed means are not cached due to the continuous nature of the trimPercent parameter.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorTrimmedMeanF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	trimPercent float32,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if trimPercent < 0 || trimPercent > 0.5 {
		panic("trimPercent must be in range [0.0, 0.5]")
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	sortedVector := core.GetSortedVector(analysis)

	// Calculate how many elements to trim from each end
	trimCount := uint64(float32(size) * trimPercent)
	if trimCount*2 >= size {
		// Trimming would remove all elements
		return 0
	}

	// Calculate mean of remaining elements
	var sum float32 = 0
	remainingCount := size - (trimCount * 2)
	for i := trimCount; i < size-trimCount; i++ {
		sum += float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, i))
	}

	return sum / float32(remainingCount)
}

/*
StatArchDescriptiveVectorTrimmedMeanF64 computes the trimmed mean (mean after removing outliers) in float64 precision.

The trimmed mean removes a specified percentage of the smallest and largest values before calculating
the mean, making it more robust to outliers than the standard mean.

Use cases:
- Robust measure of central tendency (less sensitive to outliers than regular mean)
- Quality control (removing measurement errors)
- Financial analysis (removing extreme market events)
- Data preprocessing (outlier-resistant feature engineering)

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- trimPercent must be in range [0.0, 0.5] (percentage to trim from each end)
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if trimPercent < 0 or trimPercent > 0.5
- If trimPercent = 0, returns the same as regular mean
- If trimPercent = 0.5, returns the median (if vector has even length, returns mean of two middle values)
- If trimming would remove all elements, returns 0

Note: Trimmed means are not cached due to the continuous nature of the trimPercent parameter.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorTrimmedMeanF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	trimPercent float64,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if trimPercent < 0 || trimPercent > 0.5 {
		panic("trimPercent must be in range [0.0, 0.5]")
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	sortedVector := core.GetSortedVector(analysis)

	// Calculate how many elements to trim from each end
	trimCount := uint64(float64(size) * trimPercent)
	if trimCount*2 >= size {
		// Trimming would remove all elements
		return 0
	}

	// Calculate mean of remaining elements
	var sum float64 = 0
	remainingCount := size - (trimCount * 2)
	for i := trimCount; i < size-trimCount; i++ {
		sum += float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, i))
	}

	return sum / float64(remainingCount)
}

/*
StatArchDescriptiveVectorMedianF32 computes the median of a vector in float32 precision.

Use cases:
- Measuring central tendency (robust to outliers)
- Quality control and process monitoring
- Financial computing (median income, prices)
- Outlier detection

Time complexity: O(n log n) - requires sorting the vector (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector does not need to contain numeric values in any specific range
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector has no elements
- For even-length vectors, returns average of two middle elements
- For odd-length vectors, returns middle element

The function uses the analysis structure to cache the computed median value.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorMedianF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMedianF32); ok {
		return cached.(float32)
	}

	sortedVector := core.GetSortedVector(analysis)
	length := analysis.Count
	if length == 0 {
		return 0
	}

	mid := length >> 1

	var median float32
	if length%2 == 0 {
		el1 := float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, mid-1))
		el2 := float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, mid))
		median = (el1 + el2) / 2.0
	} else {
		median = float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, mid))
	}

	core.SetCacheValue(analysis, core.StatKindMedianF32, median)
	return median
}

/*
StatArchDescriptiveVectorMedianF64 computes the median of a vector in float64 precision.

Use cases:
- Measuring central tendency (robust to outliers)
- Quality control and process monitoring
- Financial computing (median income, prices)
- Outlier detection

Time complexity: O(n log n) - requires sorting the vector (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector does not need to contain numeric values in any specific range
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector has no elements
- For even-length vectors, returns average of two middle elements
- For odd-length vectors, returns middle element

The function uses the analysis structure to cache the computed median value.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorMedianF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMedianF64); ok {
		return cached.(float64)
	}

	sortedVector := core.GetSortedVector(analysis)
	length := analysis.Count
	if length == 0 {
		return 0
	}

	mid := length >> 1

	var median float64
	if length%2 == 0 {
		el1 := float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, mid-1))
		el2 := float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, mid))
		median = (el1 + el2) / 2.0
	} else {
		median = float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, mid))
	}

	core.SetCacheValue(analysis, core.StatKindMedianF64, median)
	return median
}

/*
StatArchDescriptiveVectorPercentileF32 computes the percentile of a vector in float32 precision.

Use cases:
- Quantile analysis and distribution characterization
- Outlier detection (e.g., 95th percentile)
- Quality control (process capability analysis)
- Financial risk analysis (VaR calculations)

Time complexity: O(1) - direct index access with interpolation (vector must be pre-sorted)
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector MUST be sorted in ascending order
- Percentile must be in range [0, 100]

Edge cases:
- Returns 0 if vector has no elements
- Percentile 0 returns the minimum value
- Percentile 100 returns the maximum value
- Uses linear interpolation between adjacent values for non-integer ranks
- Panics if percentile > 100

Note: This function does not sort the vector. The caller must ensure the vector is sorted.
*/
func StatArchDescriptiveVectorPercentileF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	percentile uint8,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	// Percentiles are not cached due to the large number of possible values (0-100)
	sortedVector := core.GetSortedVector(analysis)
	length := analysis.Count
	if length == 0 {
		return 0
	}
	if percentile > 100 {
		panic("percentile > 100")
	}

	if percentile == 0 {
		return float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, 0))
	}
	if percentile == 100 {
		return float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, length-1))
	}

	if length == 1 {
		return float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, 0))
	}

	rank := float32(percentile) / 100.0 * float32(length-1)
	index := uint64(rank)
	fraction := rank - float32(index)

	v1 := float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, index))
	v2 := float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, index+1))

	return v1 + fraction*(v2-v1)
}

/*
StatArchDescriptiveVectorPercentileF64 computes the percentile of a vector in float64 precision.

Use cases:
- Quantile analysis and distribution characterization
- Outlier detection (e.g., 95th percentile)
- Quality control (process capability analysis)
- Financial risk analysis (VaR calculations)

Time complexity: O(n log n) - requires sorting the vector (first call only, then O(1) for percentile calculation)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Percentile must be in range [0, 100]
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector has no elements
- Returns the single element value if vector has exactly 1 element (any percentile)
- Percentile 0 returns the minimum value
- Percentile 100 returns the maximum value
- Uses linear interpolation between adjacent values for non-integer ranks
- Panics if percentile > 100

Note: Percentiles are not cached due to the large number of possible values (0-100).
A sorted copy of the vector is created on first use and reused for subsequent percentile calculations.
*/
func StatArchDescriptiveVectorPercentileF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	percentile uint8,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	sortedVector := core.GetSortedVector(analysis)
	length := analysis.Count
	if length == 0 {
		return 0
	}
	if percentile > 100 {
		panic("percentile > 100")
	}

	if percentile == 0 {
		return float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, 0))
	}
	if percentile == 100 {
		return float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, length-1))
	}

	if length == 1 {
		return float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, 0))
	}

	rank := float64(percentile) / 100.0 * float64(length-1)
	index := uint64(rank)
	fraction := rank - float64(index)

	v1 := float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, index))
	v2 := float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, index+1))

	return v1 + fraction*(v2-v1)
}

/*
StatArchDescriptiveVectorMode computes the most frequently occurring value in a vector.

Determinism:
If multiple values occur with the same maximum frequency (multi-modal), this function
returns the SMALLEST value among them (due to the internal sorting). This provides
consistent, deterministic results across different runs.

Time complexity: O(N log N) - dominated by the sort (first call only, then O(1) from cache)
Space complexity: O(n) - creates scratch vector on first call

Prerequisites:
- Vector does not need to be sorted
- Vector can contain any numeric type

Edge Cases:
- Vector size 0: Returns (0, 0)
- Vector size 1: Returns (value, 1)
- All values unique: Returns the smallest value in the set with an occurrence of 1
- All values identical: Returns the value with an occurrence of N

The function uses the analysis structure to cache the computed mode and occurrence values.
A scratch vector is created on first use for sorting and reused for subsequent operations.
*/
func StatArchDescriptiveVectorMode[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) (mode T, occurrence uint) {
	core.StatArchAnalysisValidateVersion(analysis)

	if cachedMode, ok := core.GetCacheValue(analysis, core.StatKindMode); ok {
		if cachedOccurrence, ok := core.GetCacheValue(analysis, core.StatKindModeOccurrence); ok {
			return cachedMode.(T), cachedOccurrence.(uint)
		}
	}

	size := analysis.Count
	if size == 0 {
		return *new(T), 0
	}
	if size == 1 {
		mode := memstruct.VectorItemGetAtUnsafe[T](analysis.Vector, 0)
		core.SetCacheValue(analysis, core.StatKindMode, mode)
		core.SetCacheValue(analysis, core.StatKindModeOccurrence, uint(1))
		return mode, 1
	}

	scratchVector := core.GetScratchVector(analysis)
	memstruct.VectorCopyFrom[T](scratchVector, analysis.Vector, 0)

	memstruct.VectorSort(scratchVector, func(a, b T) int {
		if a < b {
			return -1
		}
		if a > b {
			return 1
		}
		return 0
	})

	var currentMode T
	var currentCount uint = 0

	var bestMode T
	var maxCount uint = 0

	first := memstruct.VectorItemGetAtUnsafe[T](scratchVector, 0)
	currentMode = first
	currentCount = 1
	bestMode = first
	maxCount = 1

	for i := uint64(1); i < size; i++ {
		val := memstruct.VectorItemGetAtUnsafe[T](scratchVector, i)

		if val == currentMode {
			currentCount++
		} else {
			if currentCount > maxCount {
				maxCount = currentCount
				bestMode = currentMode
			}
			currentMode = val
			currentCount = 1
		}
	}

	if currentCount > maxCount {
		maxCount = currentCount
		bestMode = currentMode
	}

	core.SetCacheValue(analysis, core.StatKindMode, bestMode)
	core.SetCacheValue(analysis, core.StatKindModeOccurrence, maxCount)

	return bestMode, maxCount
}

/*
StatArchDescriptiveVectorVarianceF32 computes the variance of a vector in float32 precision.

Use cases:
- Measuring data spread and variability
- Quality control and process monitoring
- Risk analysis in financial computing
- Feature analysis in machine learning

Time complexity: O(n) - single pass through vector after mean calculation
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 2 elements
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements
- Returns 0 if all elements are identical (zero variance)
- Sample variance uses Bessel's correction (divides by n-1 instead of n)

The function uses the analysis structure to cache the computed variance value.
Sample and population variance are cached separately to ensure accuracy.
If the variance has already been computed for the same sample flag, it returns the cached value.
*/
func StatArchDescriptiveVectorVarianceF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindVarianceF32Sample
	} else {
		kind = core.StatKindVarianceF32Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float32)
	}

	size := analysis.Count
	if size < 2 {
		panic("not possible to compute variance with less than 2 items")
	}

	mean := StatArchDescriptiveVectorMeanF32(analysis)

	var sumSquaredDeviations float32
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		diff := float32(item) - mean
		sumSquaredDeviations += diff * diff
	}, core.StatarchDefaultStride)

	denominator := float32(size)
	if sample {
		denominator = float32(size) - 1
	}

	variance := sumSquaredDeviations / denominator

	core.SetCacheValue(analysis, kind, variance)

	return variance
}

/*
StatArchDescriptiveVectorVarianceF64 computes the variance of a vector in float64 precision.

Use cases:
- Measuring data spread and variability
- Quality control and process monitoring
- Risk analysis in financial computing
- Feature analysis in machine learning

Time complexity: O(n) - single pass through vector after mean calculation
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 2 elements
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements
- Returns 0 if all elements are identical (zero variance)
- Sample variance uses Bessel's correction (divides by n-1 instead of n)

The function uses the analysis structure to cache the computed variance value.
Sample and population variance are cached separately to ensure accuracy.
If the variance has already been computed for the same sample flag, it returns the cached value.

Numerical Stability Note: This function uses the naive variance formula which can suffer from numerical
instability for large values. For maximum numerical stability, use StatArchDescriptiveVectorAnalyzeWelford
which uses Welford's online algorithm. See README "Accuracy and Simplifications" section for details.
*/
func StatArchDescriptiveVectorVarianceF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindVarianceF64Sample
	} else {
		kind = core.StatKindVarianceF64Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float64)
	}

	size := analysis.Count
	if size < 2 {
		panic("not possible to compute variance with less than 2 items")
	}

	mean := StatArchDescriptiveVectorMeanF64(analysis)

	var sumSquaredDeviations float64
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		diff := float64(item) - mean
		sumSquaredDeviations += diff * diff
	}, core.StatarchDefaultStride)

	denominator := float64(size)
	if sample {
		denominator = float64(size) - 1
	}

	variance := sumSquaredDeviations / denominator

	core.SetCacheValue(analysis, kind, variance)

	return variance
}

/*
StatArchDescriptiveVectorStandardDeviationF32 computes the standard deviation of a vector in float32 precision.

Use cases:
- Measuring data spread (most common measure of variability)
- Quality control and process monitoring
- Risk analysis in financial computing
- Statistical process control

Time complexity: O(n) - depends on variance calculation (O(n)) plus square root (O(1))
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 2 elements
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements
- Returns 0 if all elements are identical (zero variance)
- Standard deviation is the square root of variance

The function uses the analysis structure to cache the computed standard deviation value.
Sample and population standard deviation are cached separately to ensure accuracy.
If the standard deviation has already been computed for the same sample flag, it returns the cached value.
*/
func StatArchDescriptiveVectorStandardDeviationF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindStddevF32Sample
	} else {
		kind = core.StatKindStddevF32Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float32)
	}

	variance := StatArchDescriptiveVectorVarianceF32(analysis, sample)
	stddev := foundation.Sqrt32(variance)

	core.SetCacheValue(analysis, kind, stddev)

	return stddev
}

/*
StatArchDescriptiveVectorStandardDeviationF64 computes the standard deviation of a vector in float64 precision.

Use cases:
- Measuring data spread (most common measure of variability)
- Quality control and process monitoring
- Risk analysis in financial computing
- Statistical process control

Time complexity: O(n) - depends on variance calculation (O(n)) plus square root (O(1))
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 2 elements
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements
- Returns 0 if all elements are identical (zero variance)
- Standard deviation is the square root of variance

The function uses the analysis structure to cache the computed standard deviation value.
Sample and population standard deviation are cached separately to ensure accuracy.
If the standard deviation has already been computed for the same sample flag, it returns the cached value.
*/
func StatArchDescriptiveVectorStandardDeviationF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindStddevF64Sample
	} else {
		kind = core.StatKindStddevF64Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float64)
	}

	variance := StatArchDescriptiveVectorVarianceF64(analysis, sample)
	stddev := foundation.Sqrt64(variance)

	core.SetCacheValue(analysis, kind, stddev)

	return stddev
}

/*
StatArchDescriptiveVectorRange computes the range (max - min) of a vector.

Use cases:
- Quick measure of data spread
- Outlier detection (very large ranges indicate outliers)
- Quality control (process variation)

Time complexity: O(n) - single pass through vector to find min and max
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector has only one element
- Returns 0 if all elements are identical
- Works with any numeric type

Note: Range is sensitive to outliers. Consider using IQR for more robust spread measurement.
*/
func StatArchDescriptiveVectorRange[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) T {
	core.StatArchAnalysisValidateVersion(analysis)

	var minV, maxV T
	var minCached, maxCached bool

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMin); ok {
		minV = cached.(T)
		minCached = true
	}
	if cached, ok := core.GetCacheValue(analysis, core.StatKindMax); ok {
		maxV = cached.(T)
		maxCached = true
	}

	if minCached && maxCached {
		return maxV - minV
	}

	minV, maxV = reduce.BlazeReduceVectorMinMax[T](analysis.Vector)

	core.SetCacheValue(analysis, core.StatKindMin, minV)
	core.SetCacheValue(analysis, core.StatKindMax, maxV)

	return maxV - minV
}

/*
StatArchDescriptiveVectorIQRF32 computes the interquartile range (75th - 25th percentiles) in float32 precision.

Use cases:
- Robust measure of data spread (not affected by outliers)
- Outlier detection (values beyond 1.5*IQR from quartiles)
- Quality control and process monitoring
- Box plot construction

Time complexity: O(n log n) - requires sorting the vector (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector should contain at least 4 elements for meaningful IQR
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if there are no elements in the vector
- Returns 0 if Q3 equals Q1 (no spread in middle 50% of data)
- IQR represents the spread of the middle 50% of the data

The function uses the analysis structure to cache the computed IQR value.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorIQRF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindIQRF32); ok {
		return cached.(float32)
	}

	q3 := StatArchDescriptiveVectorPercentileF32(analysis, 75)
	q1 := StatArchDescriptiveVectorPercentileF32(analysis, 25)

	iqr := q3 - q1
	core.SetCacheValue(analysis, core.StatKindIQRF32, iqr)
	return iqr
}

/*
StatArchDescriptiveVectorIQRF64 computes the interquartile range (75th - 25th percentiles) in float64 precision.

Use cases:
- Robust measure of data spread (not affected by outliers)
- Outlier detection (values beyond 1.5*IQR from quartiles)
- Quality control and process monitoring
- Box plot construction

Time complexity: O(n log n) - requires sorting the vector (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector should contain at least 4 elements for meaningful IQR
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if there are no elements in the vector
- Returns 0 if Q3 equals Q1 (no spread in middle 50% of data)
- IQR represents the spread of the middle 50% of the data

The function uses the analysis structure to cache the computed IQR value.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorIQRF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindIQRF64); ok {
		return cached.(float64)
	}

	q3 := StatArchDescriptiveVectorPercentileF64(analysis, 75)
	q1 := StatArchDescriptiveVectorPercentileF64(analysis, 25)

	iqr := q3 - q1
	core.SetCacheValue(analysis, core.StatKindIQRF64, iqr)
	return iqr
}

/*
StatArchDescriptiveVectorNormalizedIQRF32 computes the normalized interquartile range in float32 precision.

The normalized IQR is a relative measure of variability that normalizes the IQR by the sum of quartiles,
making it comparable across different scales. Formula: (Q3 - Q1) / (Q3 + Q1)

Use cases:
- Comparing variability across datasets with different scales
- Robust relative measure of spread (not affected by outliers)
- Quality control (comparing process variability across different units)
- Data normalization and standardization

Time complexity: O(n log n) - requires sorting the vector (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector should contain at least 4 elements for meaningful normalized IQR
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if there are no elements in the vector
- Returns 0 if Q3 equals Q1 (no spread in middle 50% of data)
- Panics if Q3 + Q1 equals 0 (division by zero, occurs when Q3 = -Q1)
- Normalized IQR ranges from 0 (no spread) to 1 (maximum relative spread)

The function uses the analysis structure to cache the computed normalized IQR value.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorNormalizedIQRF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindNormalizedIQRF32); ok {
		return cached.(float32)
	}

	q3 := StatArchDescriptiveVectorPercentileF32(analysis, 75)
	q1 := StatArchDescriptiveVectorPercentileF32(analysis, 25)

	denominator := q3 + q1
	if denominator == 0 {
		panic("Q3 + Q1 equals 0, cannot compute normalized IQR")
	}

	normalizedIQR := (q3 - q1) / denominator
	core.SetCacheValue(analysis, core.StatKindNormalizedIQRF32, normalizedIQR)
	return normalizedIQR
}

/*
StatArchDescriptiveVectorNormalizedIQRF64 computes the normalized interquartile range in float64 precision.

The normalized IQR is a relative measure of variability that normalizes the IQR by the sum of quartiles,
making it comparable across different scales. Formula: (Q3 - Q1) / (Q3 + Q1)

Use cases:
- Comparing variability across datasets with different scales
- Robust relative measure of spread (not affected by outliers)
- Quality control (comparing process variability across different units)
- Data normalization and standardization

Time complexity: O(n log n) - requires sorting the vector (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector should contain at least 4 elements for meaningful normalized IQR
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if there are no elements in the vector
- Returns 0 if Q3 equals Q1 (no spread in middle 50% of data)
- Panics if Q3 + Q1 equals 0 (division by zero, occurs when Q3 = -Q1)
- Normalized IQR ranges from 0 (no spread) to 1 (maximum relative spread)

The function uses the analysis structure to cache the computed normalized IQR value.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorNormalizedIQRF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindNormalizedIQRF64); ok {
		return cached.(float64)
	}

	q3 := StatArchDescriptiveVectorPercentileF64(analysis, 75)
	q1 := StatArchDescriptiveVectorPercentileF64(analysis, 25)

	denominator := q3 + q1
	if denominator == 0 {
		panic("Q3 + Q1 equals 0, cannot compute normalized IQR")
	}

	normalizedIQR := (q3 - q1) / denominator
	core.SetCacheValue(analysis, core.StatKindNormalizedIQRF64, normalizedIQR)
	return normalizedIQR
}

/*
StatArchDescriptiveVectorSkewnessF32 computes the skewness of a vector in float32 precision.

Use cases:
- Measuring asymmetry of data distribution
- Quality control (detecting process shifts)
- Financial analysis (return distribution analysis)
- Data preprocessing in machine learning

Time complexity: O(n) - single pass through vector after mean and stddev calculation
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 2 elements (population) or 3 elements (sample)
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements (population) or 3 elements (sample)
- Returns 0 if standard deviation is 0 (all values identical)
- Positive skewness indicates right tail, negative indicates left tail
- Zero skewness indicates symmetric distribution

The function uses the analysis structure to cache computed mean and standard deviation values.
*/
func StatArchDescriptiveVectorSkewnessF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindSkewnessF32Sample
	} else {
		kind = core.StatKindSkewnessF32Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float32)
	}

	size := analysis.Count

	if sample && size < 3 {
		panic("cannot compute sample skewness with less than 3 items")
	} else if size < 2 {
		panic("cannot compute skewness with less than 2 items")
	}

	standardDeviation := StatArchDescriptiveVectorStandardDeviationF32(analysis, sample)

	if standardDeviation == 0 {
		core.SetCacheValue(analysis, kind, float32(0))
		return 0
	}

	mean := StatArchDescriptiveVectorMeanF32(analysis)

	var accumulatedSkew float32 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		zScore := (float32(item) - mean) / standardDeviation
		accumulatedSkew += zScore * zScore * zScore
	}, core.StatarchDefaultStride)

	n := float32(size)
	var skewness float32
	if sample {
		correction := n / ((n - 1) * (n - 2))
		skewness = correction * accumulatedSkew
	} else {
		skewness = accumulatedSkew / n
	}

	core.SetCacheValue(analysis, kind, skewness)
	return skewness
}

/*
StatArchDescriptiveVectorSkewnessF64 computes the skewness of a vector in float64 precision.

Use cases:
- Measuring asymmetry of data distribution
- Quality control (detecting process shifts)
- Financial analysis (return distribution analysis)
- Data preprocessing in machine learning

Time complexity: O(n) - single pass through vector after mean and stddev calculation
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 2 elements (population) or 3 elements (sample)
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements (population) or 3 elements (sample)
- Returns 0 if standard deviation is 0 (all values identical)
- Positive skewness indicates right tail, negative indicates left tail
- Zero skewness indicates symmetric distribution

The function uses the analysis structure to cache computed mean and standard deviation values.
*/
func StatArchDescriptiveVectorSkewnessF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindSkewnessF64Sample
	} else {
		kind = core.StatKindSkewnessF64Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float64)
	}

	size := analysis.Count

	if sample && size < 3 {
		panic("cannot compute sample skewness with less than 3 items")
	} else if size < 2 {
		panic("cannot compute skewness with less than 2 items")
	}

	standardDeviation := StatArchDescriptiveVectorStandardDeviationF64(analysis, sample)

	if standardDeviation == 0 {
		core.SetCacheValue(analysis, kind, float64(0))
		return 0
	}

	mean := StatArchDescriptiveVectorMeanF64(analysis)

	var accumulatedSkew float64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		zScore := (float64(item) - mean) / standardDeviation
		accumulatedSkew += zScore * zScore * zScore
	}, core.StatarchDefaultStride)

	n := float64(size)
	var skewness float64
	if sample {
		correction := n / ((n - 1) * (n - 2))
		skewness = correction * accumulatedSkew
	} else {
		skewness = accumulatedSkew / n
	}

	core.SetCacheValue(analysis, kind, skewness)
	return skewness
}

/*
StatArchDescriptiveVectorKurtosisF32 computes the excess kurtosis of a vector in float32 precision.

Excess kurtosis is kurtosis minus 3, which centers the measure at 0 for normal distributions.
This is the most commonly used form of kurtosis in statistical analysis.

Use cases:
- Measuring tail heaviness of data distribution
- Financial risk analysis (fat tails indicate higher risk)
- Quality control (detecting process changes)
- Distribution shape analysis

Time complexity: O(n) - single pass through vector after mean and stddev calculation
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 3 elements (population) or 4 elements (sample)
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Standard deviation must not be 0
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 3 elements (population) or 4 elements (sample)
- Panics if standard deviation is 0 (cannot compute z-scores)
- Returns 0 for normal distribution (excess kurtosis = kurtosis - 3)
- Positive excess kurtosis indicates heavy tails, negative indicates light tails

The function uses the analysis structure to cache computed mean and standard deviation values.
Sample and population excess kurtosis are cached separately to ensure accuracy.
*/
func StatArchDescriptiveVectorKurtosisF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindKurtosisF32Sample
	} else {
		kind = core.StatKindKurtosisF32Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float32)
	}

	size := analysis.Count

	if sample && size < 4 {
		panic("cannot compute sample kurtosis with less than 4 items")
	} else if size < 3 {
		panic("cannot compute population kurtosis with less than 3 items")
	}

	standardDeviation := StatArchDescriptiveVectorStandardDeviationF32(analysis, sample)

	if standardDeviation == 0 {
		panic("Standard deviation is 0, cannot compute kurtosis.")
	}

	mean := StatArchDescriptiveVectorMeanF32(analysis)
	sd64 := float64(standardDeviation)
	mean64 := float64(mean)

	var accumulatedKurtosis float64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		zScore := (float64(item) - mean64) / sd64
		z2 := zScore * zScore
		accumulatedKurtosis += z2 * z2
	}, core.StatarchDefaultStride)

	n := float64(size)
	var kurtosis float32
	if sample {
		term1 := (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
		term2 := (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3))
		kurtosis = float32((term1 * accumulatedKurtosis) - term2)
	} else {
		kurtosis = float32(accumulatedKurtosis/n) - 3
	}

	core.SetCacheValue(analysis, kind, kurtosis)
	return kurtosis
}

/*
StatArchDescriptiveVectorKurtosisF64 computes the excess kurtosis of a vector in float64 precision.

Excess kurtosis is kurtosis minus 3, which centers the measure at 0 for normal distributions.
This is the most commonly used form of kurtosis in statistical analysis.

Use cases:
- Measuring tail heaviness of data distribution
- Financial risk analysis (fat tails indicate higher risk)
- Quality control (detecting process changes)
- Distribution shape analysis

Time complexity: O(n) - single pass through vector after mean and stddev calculation
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 3 elements (population) or 4 elements (sample)
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Standard deviation must not be 0
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 3 elements (population) or 4 elements (sample)
- Panics if standard deviation is 0 (cannot compute z-scores)
- Returns 0 for normal distribution (excess kurtosis = kurtosis - 3)
- Positive excess kurtosis indicates heavy tails, negative indicates light tails

The function uses the analysis structure to cache computed mean and standard deviation values.
Sample and population excess kurtosis are cached separately to ensure accuracy.
*/
func StatArchDescriptiveVectorKurtosisF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindKurtosisF64Sample
	} else {
		kind = core.StatKindKurtosisF64Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float64)
	}

	size := analysis.Count

	if sample && size < 4 {
		panic("cannot compute sample kurtosis with less than 4 items")
	} else if size < 3 {
		panic("cannot compute population kurtosis with less than 3 items")
	}

	standardDeviation := StatArchDescriptiveVectorStandardDeviationF64(analysis, sample)

	if standardDeviation == 0 {
		panic("Standard deviation is 0, cannot compute kurtosis.")
	}

	mean := StatArchDescriptiveVectorMeanF64(analysis)

	var accumulatedKurtosis float64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		zScore := (float64(item) - mean) / standardDeviation
		z2 := zScore * zScore
		accumulatedKurtosis += z2 * z2
	}, core.StatarchDefaultStride)

	n := float64(size)
	var kurtosis float64
	if sample {
		term1 := (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
		term2 := (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3))
		kurtosis = (term1 * accumulatedKurtosis) - term2
	} else {
		kurtosis = accumulatedKurtosis/n - 3
	}

	core.SetCacheValue(analysis, kind, kurtosis)
	return kurtosis
}

/*
StatArchDescriptiveVectorMeanAbsoluteDeviationF32 computes the mean absolute deviation in float32 precision.

Use cases:
- Robust measure of spread (less sensitive to outliers than standard deviation)
- Quality control and process monitoring
- Financial risk analysis

Time complexity: O(n) - single pass through vector after mean calculation
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if no elements in the vector
- Returns 0 if all elements are identical
- Formula: Σ|x - μ| / N

The function uses the analysis structure to cache the computed mean value.
*/
func StatArchDescriptiveVectorMeanAbsoluteDeviationF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMeanAbsoluteDeviationF32); ok {
		return cached.(float32)
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	mean := StatArchDescriptiveVectorMeanF32(analysis)
	mean64 := float64(mean)

	var accumulatedDeviation float64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		distance := float64(item) - mean64
		if distance < 0 {
			distance = -distance
		}
		accumulatedDeviation += distance
	}, core.StatarchDefaultStride)

	mad := float32(accumulatedDeviation / float64(size))
	core.SetCacheValue(analysis, core.StatKindMeanAbsoluteDeviationF32, mad)
	return mad
}

/*
StatArchDescriptiveVectorMeanAbsoluteDeviationF64 computes the mean absolute deviation in float64 precision.

Use cases:
- Robust measure of spread (less sensitive to outliers than standard deviation)
- Quality control and process monitoring
- Financial risk analysis

Time complexity: O(n) - single pass through vector after mean calculation
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if no elements in the vector
- Returns 0 if all elements are identical
- Formula: Σ|x - μ| / N

The function uses the analysis structure to cache the computed mean value.
*/
func StatArchDescriptiveVectorMeanAbsoluteDeviationF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMeanAbsoluteDeviationF64); ok {
		return cached.(float64)
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	mean := StatArchDescriptiveVectorMeanF64(analysis)

	var accumulatedDeviation float64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		distance := float64(item) - mean
		if distance < 0 {
			distance = -distance
		}
		accumulatedDeviation += distance
	}, core.StatarchDefaultStride)

	mad := accumulatedDeviation / float64(size)
	core.SetCacheValue(analysis, core.StatKindMeanAbsoluteDeviationF64, mad)
	return mad
}

/*
StatArchDescriptiveVectorMedianAbsoluteDeviationF32 computes the median absolute deviation in float32 precision.

Use cases:
- Highly robust measure of spread (resistant to outliers)
- Outlier detection
- Robust statistics and non-parametric analysis

Time complexity: O(n log n) - requires sorting the vector and absolute deviations (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy and scratch vector on first call

Prerequisites:
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if there are no elements in the vector
- Returns scaled MAD (multiplied by 1.4826) for consistency with normal distribution standard deviation

The function uses the analysis structure to cache the computed MAD value.
A sorted copy of the vector and scratch vectors are created on first use and reused for subsequent operations.

Note: The function computes median of absolute deviations from the median, then scales by 1.4826
to make it comparable to standard deviation for normally distributed data.
*/
func StatArchDescriptiveVectorMedianAbsoluteDeviationF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMedianAbsoluteDeviationF32); ok {
		return cached.(float32)
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	median := StatArchDescriptiveVectorMedianF32(analysis)
	sortedVector := core.GetSortedVector(analysis)
	scratchVector := core.GetScratchVectorF32(analysis)

	// Use scratch vector as float32 vector for absolute deviations
	if memstruct.VectorCapacityGet[float32](scratchVector) < size {
		panic("scratchVector capacity is smaller than input vector")
	}

	var idx uint64 = 0
	memstruct.VectorUnaryReadOnlyExecute(sortedVector, func(item T) {
		diff := float32(item) - median
		if diff < 0 {
			diff = -diff
		}
		memstruct.VectorSetAtUnsafe(scratchVector, idx, diff)
		idx++
	}, 1)

	memstruct.VectorSort(scratchVector, func(a, b float32) int {
		if a < b {
			return -1
		}
		if a > b {
			return 1
		}
		return 0
	})

	// Compute median of absolute deviations
	mid := size >> 1
	var madMedian float32
	if size%2 == 0 {
		el1 := memstruct.VectorItemGetAtUnsafe[float32](scratchVector, mid-1)
		el2 := memstruct.VectorItemGetAtUnsafe[float32](scratchVector, mid)
		madMedian = (el1 + el2) / 2.0
	} else {
		madMedian = memstruct.VectorItemGetAtUnsafe[float32](scratchVector, mid)
	}

	mad := madMedian * 1.4826
	core.SetCacheValue(analysis, core.StatKindMedianAbsoluteDeviationF32, mad)
	return mad
}

/*
StatArchDescriptiveVectorMedianAbsoluteDeviationF64 computes the median absolute deviation in float64 precision.

Use cases:
- Highly robust measure of spread (resistant to outliers)
- Outlier detection
- Robust statistics and non-parametric analysis

Time complexity: O(n log n) - requires sorting the vector and absolute deviations (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy and scratch vector on first call

Prerequisites:
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if there are no elements in the vector
- Returns scaled MAD (multiplied by 1.4826) for consistency with normal distribution standard deviation

The function uses the analysis structure to cache the computed MAD value.
A sorted copy of the vector and scratch vectors are created on first use and reused for subsequent operations.

Note: The function computes median of absolute deviations from the median, then scales by 1.4826
to make it comparable to standard deviation for normally distributed data.
*/
func StatArchDescriptiveVectorMedianAbsoluteDeviationF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMedianAbsoluteDeviationF64); ok {
		return cached.(float64)
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	median := StatArchDescriptiveVectorMedianF64(analysis)
	sortedVector := core.GetSortedVector(analysis)
	scratchVector := core.GetScratchVectorF64(analysis)

	// Use scratch vector as float64 vector for absolute deviations
	if memstruct.VectorCapacityGet[float64](scratchVector) < size {
		panic("scratchVector capacity mismatch")
	}

	var idx uint64 = 0
	memstruct.VectorUnaryReadOnlyExecute(sortedVector, func(item T) {
		diff := float64(item) - median
		if diff < 0 {
			diff = -diff
		}
		memstruct.VectorSetAtUnsafe(scratchVector, idx, diff)
		idx++
	}, 1)

	memstruct.VectorSort(scratchVector, func(a, b float64) int {
		if a < b {
			return -1
		}
		if a > b {
			return 1
		}
		return 0
	})

	// Compute median of absolute deviations
	mid := size >> 1
	var madMedian float64
	if size%2 == 0 {
		el1 := memstruct.VectorItemGetAtUnsafe[float64](scratchVector, mid-1)
		el2 := memstruct.VectorItemGetAtUnsafe[float64](scratchVector, mid)
		madMedian = (el1 + el2) / 2.0
	} else {
		madMedian = memstruct.VectorItemGetAtUnsafe[float64](scratchVector, mid)
	}

	mad := madMedian * 1.4826
	core.SetCacheValue(analysis, core.StatKindMedianAbsoluteDeviationF64, mad)
	return mad
}

/*
StatArchDescriptiveVectorCoefficientVariantF32 computes the coefficient of variation in float32 precision.

Use cases:
- Normalized measure of variability (relative to mean)
- Comparing variability across different scales
- Quality control (process capability)
- Financial analysis (risk-adjusted returns)

Time complexity: O(n) - depends on mean and standard deviation calculation
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 2 elements
- Mean must not be 0 (division by zero)
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Vector does not need to be sorted

Edge cases:
- Panics if mean is 0 (division by zero)
- Returns 0 if standard deviation is 0 (no variability)
- Formula: CV = σ / μ (standard deviation divided by mean)

The function uses the analysis structure to cache computed mean and standard deviation values.
*/
func StatArchDescriptiveVectorCoefficientVariantF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindCoefficientVariationF32Sample
	} else {
		kind = core.StatKindCoefficientVariationF32Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float32)
	}

	standardDeviation := StatArchDescriptiveVectorStandardDeviationF32(analysis, sample)
	mean := StatArchDescriptiveVectorMeanF32(analysis)

	if mean == 0 {
		panic("mean is 0, cannot compute coefficient of variation")
	}

	cv := standardDeviation / mean
	core.SetCacheValue(analysis, kind, cv)
	return cv
}

/*
StatArchDescriptiveVectorCoefficientVariantF64 computes the coefficient of variation in float64 precision.

Use cases:
- Normalized measure of variability (relative to mean)
- Comparing variability across different scales
- Quality control (process capability)
- Financial analysis (risk-adjusted returns)

Time complexity: O(n) - depends on mean and standard deviation calculation
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 2 elements
- Mean must not be 0 (division by zero)
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population
- Vector does not need to be sorted

Edge cases:
- Panics if mean is 0 (division by zero)
- Returns 0 if standard deviation is 0 (no variability)
- Formula: CV = σ / μ (standard deviation divided by mean)

The function uses the analysis structure to cache computed mean and standard deviation values.
*/
func StatArchDescriptiveVectorCoefficientVariantF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindCoefficientVariationF64Sample
	} else {
		kind = core.StatKindCoefficientVariationF64Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float64)
	}

	standardDeviation := StatArchDescriptiveVectorStandardDeviationF64(analysis, sample)
	mean := StatArchDescriptiveVectorMeanF64(analysis)

	if mean == 0 {
		panic("mean is 0, cannot compute coefficient of variation")
	}

	cv := standardDeviation / mean
	core.SetCacheValue(analysis, kind, cv)
	return cv
}

/*
StatArchDescriptiveVectorAnalyzeWelford computes Mean, Variance, Skewness,
and Kurtosis in a single pass and updates the analysis cache.

This is the most efficient way to get a full distribution profile, minimizing
memory bandwidth by iterating over the vector exactly once. It uses Welford's
online algorithm for superior numerical stability.

Time complexity: O(n)
Space complexity: O(1) accumulators + O(k) cache entries

Parameters:
- analysis: The analysis structure containing the vector and cache.
- sample: If true, computes unbiased sample statistics. If false, population.
*/
func StatArchDescriptiveVectorAnalyzeWelford[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) (mean, variance, skewness, kurtosis float64) {
	core.StatArchAnalysisValidateVersion(analysis)

	size := analysis.Count
	if size == 0 {
		return 0, 0, 0, 0
	}

	var n float64 = 0
	var M1, M2, M3, M4 float64 = 0, 0, 0, 0

	// Single pass over the memory
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		val := float64(item)
		n1 := n
		n = n + 1

		delta := val - M1
		delta_n := delta / n
		delta_n2 := delta_n * delta_n
		term1 := delta * delta_n * n1

		M1 += delta_n
		M4 += term1*delta_n2*(n*n-3*n+3) + 6*delta_n2*M2 - 4*delta_n*M3
		M3 += term1*delta_n*(n-2) - 3*delta_n*M2
		M2 += term1
	}, 1)

	mean = M1

	// Precision handling: store both F32 and F64 means in cache
	core.SetCacheValue(analysis, core.StatKindMeanF64, mean)
	core.SetCacheValue(analysis, core.StatKindMeanF32, float32(mean))

	if n < 2 {
		// Cache zero variance and stddev for edge case
		zeroVar := float64(0)
		zeroStddev := float64(0)
		if sample {
			core.SetCacheValue(analysis, core.StatKindVarianceF64Sample, zeroVar)
			core.SetCacheValue(analysis, core.StatKindVarianceF32Sample, float32(zeroVar))
			core.SetCacheValue(analysis, core.StatKindStddevF64Sample, zeroStddev)
			core.SetCacheValue(analysis, core.StatKindStddevF32Sample, float32(zeroStddev))
		} else {
			core.SetCacheValue(analysis, core.StatKindVarianceF64Population, zeroVar)
			core.SetCacheValue(analysis, core.StatKindVarianceF32Population, float32(zeroVar))
			core.SetCacheValue(analysis, core.StatKindStddevF64Population, zeroStddev)
			core.SetCacheValue(analysis, core.StatKindStddevF32Population, float32(zeroStddev))
		}
		return mean, 0, 0, 0
	}

	// 1. Variance
	if sample {
		variance = M2 / (n - 1)
		core.SetCacheValue(analysis, core.StatKindVarianceF64Sample, variance)
		core.SetCacheValue(analysis, core.StatKindVarianceF32Sample, float32(variance))
	} else {
		variance = M2 / n
		core.SetCacheValue(analysis, core.StatKindVarianceF64Population, variance)
		core.SetCacheValue(analysis, core.StatKindVarianceF32Population, float32(variance))
	}

	// Cache standard deviation
	stddev := foundation.Sqrt64(variance)
	if sample {
		core.SetCacheValue(analysis, core.StatKindStddevF64Sample, stddev)
		core.SetCacheValue(analysis, core.StatKindStddevF32Sample, float32(stddev))
	} else {
		core.SetCacheValue(analysis, core.StatKindStddevF64Population, stddev)
		core.SetCacheValue(analysis, core.StatKindStddevF32Population, float32(stddev))
	}

	if M2 == 0 {
		// Cache zero skewness and kurtosis for edge case
		zeroSkew := float64(0)
		zeroKurt := float64(0)
		if sample {
			core.SetCacheValue(analysis, core.StatKindSkewnessF64Sample, zeroSkew)
			core.SetCacheValue(analysis, core.StatKindSkewnessF32Sample, float32(zeroSkew))
			core.SetCacheValue(analysis, core.StatKindKurtosisF64Sample, zeroKurt)
			core.SetCacheValue(analysis, core.StatKindKurtosisF32Sample, float32(zeroKurt))
		} else {
			core.SetCacheValue(analysis, core.StatKindSkewnessF64Population, zeroSkew)
			core.SetCacheValue(analysis, core.StatKindSkewnessF32Population, float32(zeroSkew))
			core.SetCacheValue(analysis, core.StatKindKurtosisF64Population, zeroKurt)
			core.SetCacheValue(analysis, core.StatKindKurtosisF32Population, float32(zeroKurt))
		}
		return mean, variance, 0, 0
	}

	// 2. Skewness
	if sample {
		if n >= 3 {
			// Unbiased Sample Skewness
			skewness = (foundation.Sqrt64(n*(n-1)) / (n - 2)) * (M3 / foundation.Pow64(M2, 1.5))
			core.SetCacheValue(analysis, core.StatKindSkewnessF64Sample, skewness)
			core.SetCacheValue(analysis, core.StatKindSkewnessF32Sample, float32(skewness))
		}
	} else {
		// Population Skewness
		skewness = M3 / foundation.Pow64(M2/n, 1.5) / n
		core.SetCacheValue(analysis, core.StatKindSkewnessF64Population, skewness)
		core.SetCacheValue(analysis, core.StatKindSkewnessF32Population, float32(skewness))
	}

	// 3. Kurtosis (Excess)
	if sample {
		if n >= 4 {
			term1 := (n * (n + 1) * (n - 1)) / ((n - 2) * (n - 3))
			term2 := (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3))
			kurtosis = (term1 * M4 / (M2 * M2)) - term2
			core.SetCacheValue(analysis, core.StatKindKurtosisF64Sample, kurtosis)
			core.SetCacheValue(analysis, core.StatKindKurtosisF32Sample, float32(kurtosis))
		}
	} else {
		kurtosis = (n * M4 / (M2 * M2)) - 3
		core.SetCacheValue(analysis, core.StatKindKurtosisF64Population, kurtosis)
		core.SetCacheValue(analysis, core.StatKindKurtosisF32Population, float32(kurtosis))
	}

	return mean, variance, skewness, kurtosis
}

/*
StatArchDescriptiveVectorStandardErrorMeanF32 computes the standard error of the mean (SEM) in float32 precision.

The standard error of the mean measures the precision of the sample mean as an estimate of the population mean.
It represents the standard deviation of the sampling distribution of the mean.

Use cases:
- Confidence interval construction for the mean
- Hypothesis testing (t-tests, z-tests)
- Sample size determination
- Quality control (process capability analysis)
- Scientific reporting (error bars in graphs)

Time complexity: O(1) - depends on standard deviation calculation (O(n)) plus square root (O(1))
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 2 elements
- If sample=true, vector represents a sample from a larger population (uses sample standard deviation)
- If sample=false, vector represents the entire population (uses population standard deviation)
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements (inherited from standard deviation calculation)
- Returns 0 if all elements are identical (zero standard deviation)
- Formula: SEM = σ / √n where σ is standard deviation and n is sample size

The function uses the analysis structure to cache the computed SEM value.
Sample and population SEM are cached separately to ensure accuracy.
If the SEM has already been computed for the same sample flag, it returns the cached value.
*/
func StatArchDescriptiveVectorStandardErrorMeanF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindStandardErrorMeanF32Sample
	} else {
		kind = core.StatKindStandardErrorMeanF32Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float32)
	}

	standardDeviation := StatArchDescriptiveVectorStandardDeviationF32(analysis, sample)
	sem := standardDeviation / foundation.Sqrt32(float32(analysis.Count))

	core.SetCacheValue(analysis, kind, sem)
	return sem
}

/*
StatArchDescriptiveVectorStandardErrorMeanF64 computes the standard error of the mean (SEM) in float64 precision.

The standard error of the mean measures the precision of the sample mean as an estimate of the population mean.
It represents the standard deviation of the sampling distribution of the mean.

Use cases:
- Confidence interval construction for the mean
- Hypothesis testing (t-tests, z-tests)
- Sample size determination
- Quality control (process capability analysis)
- Scientific reporting (error bars in graphs)

Time complexity: O(1) - depends on standard deviation calculation (O(n)) plus square root (O(1))
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 2 elements
- If sample=true, vector represents a sample from a larger population (uses sample standard deviation)
- If sample=false, vector represents the entire population (uses population standard deviation)
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements (inherited from standard deviation calculation)
- Returns 0 if all elements are identical (zero standard deviation)
- Formula: SEM = σ / √n where σ is standard deviation and n is sample size

The function uses the analysis structure to cache the computed SEM value.
Sample and population SEM are cached separately to ensure accuracy.
If the SEM has already been computed for the same sample flag, it returns the cached value.
*/
func StatArchDescriptiveVectorStandardErrorMeanF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	var kind core.StatKind
	if sample {
		kind = core.StatKindStandardErrorMeanF64Sample
	} else {
		kind = core.StatKindStandardErrorMeanF64Population
	}

	if cached, ok := core.GetCacheValue(analysis, kind); ok {
		return cached.(float64)
	}

	standardDeviation := StatArchDescriptiveVectorStandardDeviationF64(analysis, sample)
	sem := standardDeviation / foundation.Sqrt64(float64(analysis.Count))

	core.SetCacheValue(analysis, kind, sem)
	return sem
}

/*
StatArchDescriptiveVectorZScoreNormalizeF32 normalizes a vector to z-scores in float32 precision.

Z-score normalization (standardization) transforms each value to have a mean of 0 and standard deviation of 1.
Formula: z = (x - μ) / σ where μ is the mean and σ is the standard deviation.

Use cases:
- Data preprocessing for machine learning (feature scaling)
- Comparing values across different scales
- Outlier detection (values beyond ±2 or ±3 standard deviations)
- Statistical analysis requiring standardized data

Time complexity: O(n) - single pass through vector after mean and stddev calculation
Space complexity: O(1) - only local variables used (destination vector must be pre-allocated)

Prerequisites:
- Vector must contain at least 2 elements
- If sample=true, vector represents a sample from a larger population (uses sample standard deviation)
- If sample=false, vector represents the entire population (uses population standard deviation)
- destinationVector must have the same capacity as the source vector
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements (inherited from standard deviation calculation)
- Panics if standard deviation is 0 (all elements identical, division by zero)
- Panics if destinationVector capacity doesn't match source vector capacity
- After normalization, the mean of z-scores will be 0 and standard deviation will be 1

The function uses the analysis structure to access cached mean and standard deviation values.
*/
func StatArchDescriptiveVectorZScoreNormalizeF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
	destinationVector memcore.MarkRaw,
) {
	size := analysis.Count
	if size < 2 {
		panic("cannot compute z-scores with less than 2 elements")
	}

	destCapacity := memstruct.VectorCapacityGet[float32](destinationVector)
	if destCapacity < size {
		panic("destinationVector capacity is smaller than source vector size")
	}

	mean := StatArchDescriptiveVectorMeanF32(analysis)
	standardDeviation := StatArchDescriptiveVectorStandardDeviationF32(analysis, sample)

	if standardDeviation == 0 {
		panic("standard deviation is 0, cannot compute z-scores (all values are identical)")
	}

	var idx uint64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		zScore := (float32(item) - mean) / standardDeviation
		memstruct.VectorSetAtUnsafe(destinationVector, idx, zScore)
		idx++
	}, core.StatarchDefaultStride)
}

/*
StatArchDescriptiveVectorZScoreNormalizeF64 normalizes a vector to z-scores in float64 precision.

Z-score normalization (standardization) transforms each value to have a mean of 0 and standard deviation of 1.
Formula: z = (x - μ) / σ where μ is the mean and σ is the standard deviation.

Use cases:
- Data preprocessing for machine learning (feature scaling)
- Comparing values across different scales
- Outlier detection (values beyond ±2 or ±3 standard deviations)
- Statistical analysis requiring standardized data

Time complexity: O(n) - single pass through vector after mean and stddev calculation
Space complexity: O(1) - only local variables used (destination vector must be pre-allocated)

Prerequisites:
- Vector must contain at least 2 elements
- If sample=true, vector represents a sample from a larger population (uses sample standard deviation)
- If sample=false, vector represents the entire population (uses population standard deviation)
- destinationVector must have the same capacity as the source vector
- Vector does not need to be sorted

Edge cases:
- Panics if vector has less than 2 elements (inherited from standard deviation calculation)
- Panics if standard deviation is 0 (all elements identical, division by zero)
- Panics if destinationVector capacity doesn't match source vector capacity
- After normalization, the mean of z-scores will be 0 and standard deviation will be 1

The function uses the analysis structure to access cached mean and standard deviation values.
*/
func StatArchDescriptiveVectorZScoreNormalizeF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
	destinationVector memcore.MarkRaw,
) {
	size := analysis.Count
	if size < 2 {
		panic("cannot compute z-scores with less than 2 elements")
	}

	destCapacity := memstruct.VectorCapacityGet[float64](destinationVector)
	if destCapacity < size {
		panic("destinationVector capacity is smaller than source vector size")
	}

	mean := StatArchDescriptiveVectorMeanF64(analysis)
	standardDeviation := StatArchDescriptiveVectorStandardDeviationF64(analysis, sample)

	if standardDeviation == 0 {
		panic("standard deviation is 0, cannot compute z-scores (all values are identical)")
	}

	var idx uint64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysis.Vector, func(item T) {
		zScore := (float64(item) - mean) / standardDeviation
		memstruct.VectorSetAtUnsafe(destinationVector, idx, zScore)
		idx++
	}, core.StatarchDefaultStride)
}

/*
StatArchDescriptiveVectorWinsorizedMeanF32 computes the winsorized mean in float32 precision.

Winsorization replaces extreme values with the nearest non-extreme value, then computes the mean.
This makes it more robust to outliers than the regular mean while using all data points.

Use cases:
- Robust measure of central tendency (less sensitive to outliers than mean, more efficient than trimmed mean)
- Quality control (handling measurement errors)
- Financial analysis (removing extreme market events)
- Data preprocessing (outlier-resistant feature engineering)

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- trimPercent must be in range [0.0, 0.5] (percentage to winsorize from each end)
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if trimPercent < 0 or trimPercent > 0.5
- If trimPercent = 0, returns the same as regular mean
- If trimPercent = 0.5, returns the median (if vector has even length, returns mean of two middle values)
- If winsorization would affect all elements, returns the median

Note: Winsorized means are not cached due to the continuous nature of the trimPercent parameter.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorWinsorizedMeanF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	trimPercent float32,
) float32 {
	if trimPercent < 0 || trimPercent > 0.5 {
		panic("trimPercent must be in range [0.0, 0.5]")
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	sortedVector := core.GetSortedVector(analysis)

	// Calculate how many elements to winsorize from each end
	trimCount := uint64(float32(size) * trimPercent)
	if trimCount*2 >= size {
		// Winsorization would affect all elements, return median
		return StatArchDescriptiveVectorMedianF32(analysis)
	}

	// Get the boundary values
	lowerBound := float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, trimCount))
	upperBound := float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, size-trimCount-1))

	// Compute winsorized mean: replace extreme values with bounds, then compute mean
	var sum float32 = 0
	for i := uint64(0); i < size; i++ {
		val := float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, i))
		if i < trimCount {
			sum += lowerBound
		} else if i >= size-trimCount {
			sum += upperBound
		} else {
			sum += val
		}
	}

	return sum / float32(size)
}

/*
StatArchDescriptiveVectorWinsorizedMeanF64 computes the winsorized mean in float64 precision.

Winsorization replaces extreme values with the nearest non-extreme value, then computes the mean.
This makes it more robust to outliers than the regular mean while using all data points.

Use cases:
- Robust measure of central tendency (less sensitive to outliers than mean, more efficient than trimmed mean)
- Quality control (handling measurement errors)
- Financial analysis (removing extreme market events)
- Data preprocessing (outlier-resistant feature engineering)

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- trimPercent must be in range [0.0, 0.5] (percentage to winsorize from each end)
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if trimPercent < 0 or trimPercent > 0.5
- If trimPercent = 0, returns the same as regular mean
- If trimPercent = 0.5, returns the median (if vector has even length, returns mean of two middle values)
- If winsorization would affect all elements, returns the median

Note: Winsorized means are not cached due to the continuous nature of the trimPercent parameter.
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorWinsorizedMeanF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	trimPercent float64,
) float64 {
	if trimPercent < 0 || trimPercent > 0.5 {
		panic("trimPercent must be in range [0.0, 0.5]")
	}

	size := analysis.Count
	if size == 0 {
		return 0
	}

	sortedVector := core.GetSortedVector(analysis)

	// Calculate how many elements to winsorize from each end
	trimCount := uint64(float64(size) * trimPercent)
	if trimCount*2 >= size {
		// Winsorization would affect all elements, return median
		return StatArchDescriptiveVectorMedianF64(analysis)
	}

	// Get the boundary values
	lowerBound := float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, trimCount))
	upperBound := float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, size-trimCount-1))

	// Compute winsorized mean: replace extreme values with bounds, then compute mean
	var sum float64 = 0
	for i := uint64(0); i < size; i++ {
		val := float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, i))
		if i < trimCount {
			sum += lowerBound
		} else if i >= size-trimCount {
			sum += upperBound
		} else {
			sum += val
		}
	}

	return sum / float64(size)
}

/*
StatArchDescriptiveVectorMidRangeF32 computes the mid-range (average of min and max) in float32 precision.

The mid-range is a simple measure of central tendency that is the midpoint between the minimum and maximum values.

Use cases:
- Quick estimate of central tendency
- Range center identification
- Simple data characterization

Time complexity: O(1) - uses cached min and max values
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty
- Formula: (min + max) / 2

The function uses the analysis structure to cache the computed mid-range value.
It uses cached min and max values from the analysis structure.
*/
func StatArchDescriptiveVectorMidRangeF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMidRangeF32); ok {
		return cached.(float32)
	}

	// Get min and max from cache or compute via Range
	var minV, maxV T
	if cachedMin, ok := core.GetCacheValue(analysis, core.StatKindMin); ok {
		minV = cachedMin.(T)
	} else {
		// Compute range which also caches min/max
		StatArchDescriptiveVectorRange(analysis)
		cachedMin, _ := core.GetCacheValue(analysis, core.StatKindMin)
		minV = cachedMin.(T)
	}
	if cachedMax, ok := core.GetCacheValue(analysis, core.StatKindMax); ok {
		maxV = cachedMax.(T)
	} else {
		// Compute range which also caches min/max
		StatArchDescriptiveVectorRange(analysis)
		cachedMax, _ := core.GetCacheValue(analysis, core.StatKindMax)
		maxV = cachedMax.(T)
	}

	midRange := (float32(minV) + float32(maxV)) / 2.0
	core.SetCacheValue(analysis, core.StatKindMidRangeF32, midRange)
	return midRange
}

/*
StatArchDescriptiveVectorMidRangeF64 computes the mid-range (average of min and max) in float64 precision.

The mid-range is a simple measure of central tendency that is the midpoint between the minimum and maximum values.

Use cases:
- Quick estimate of central tendency
- Range center identification
- Simple data characterization

Time complexity: O(1) - uses cached min and max values
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty
- Formula: (min + max) / 2

The function uses the analysis structure to cache the computed mid-range value.
It uses cached min and max values from the analysis structure.
*/
func StatArchDescriptiveVectorMidRangeF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindMidRangeF64); ok {
		return cached.(float64)
	}

	// Get min and max from cache or compute via Range
	var minV, maxV T
	if cachedMin, ok := core.GetCacheValue(analysis, core.StatKindMin); ok {
		minV = cachedMin.(T)
	} else {
		// Compute range which also caches min/max
		StatArchDescriptiveVectorRange(analysis)
		cachedMin, _ := core.GetCacheValue(analysis, core.StatKindMin)
		minV = cachedMin.(T)
	}
	if cachedMax, ok := core.GetCacheValue(analysis, core.StatKindMax); ok {
		maxV = cachedMax.(T)
	} else {
		// Compute range which also caches min/max
		StatArchDescriptiveVectorRange(analysis)
		cachedMax, _ := core.GetCacheValue(analysis, core.StatKindMax)
		maxV = cachedMax.(T)
	}

	midRange := (float64(minV) + float64(maxV)) / 2.0
	core.SetCacheValue(analysis, core.StatKindMidRangeF64, midRange)
	return midRange
}

// FiveNumberSummaryF32 holds the five-number summary statistics in float32 precision.
type FiveNumberSummaryF32 struct {
	Min    float32
	Q1     float32
	Median float32
	Q3     float32
	Max    float32
}

// FiveNumberSummaryF64 holds the five-number summary statistics in float64 precision.
type FiveNumberSummaryF64 struct {
	Min    float64
	Q1     float64
	Median float64
	Q3     float64
	Max    float64
}

/*
StatArchDescriptiveVectorFiveNumberSummaryF32 computes the five-number summary in float32 precision.

The five-number summary consists of: Minimum, First Quartile (Q1), Median, Third Quartile (Q3), and Maximum.
This provides a comprehensive overview of the data distribution.

Use cases:
- Box plot construction
- Quick data distribution overview
- Outlier detection (using IQR = Q3 - Q1)
- Statistical reporting

Time complexity: O(n log n) - requires sorting the vector (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector should contain at least 4 elements for meaningful quartiles
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns zero values if vector is empty
- For small vectors, quartiles may equal min/max/median

The function uses the analysis structure to efficiently compute and cache all five values.
All individual statistics (Min, Max, Median, Q1, Q3) are cached for reuse by other functions.
*/
func StatArchDescriptiveVectorFiveNumberSummaryF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) FiveNumberSummaryF32 {
	// Use cached values where available, compute others
	// Ensure min/max are cached by computing range if needed
	StatArchDescriptiveVectorRange(analysis)
	cachedMin, _ := core.GetCacheValue(analysis, core.StatKindMin)
	cachedMax, _ := core.GetCacheValue(analysis, core.StatKindMax)
	min := float32(cachedMin.(T))
	max := float32(cachedMax.(T))
	median := StatArchDescriptiveVectorMedianF32(analysis)
	q1 := StatArchDescriptiveVectorPercentileF32(analysis, 25)
	q3 := StatArchDescriptiveVectorPercentileF32(analysis, 75)

	return FiveNumberSummaryF32{
		Min:    min,
		Q1:     q1,
		Median: median,
		Q3:     q3,
		Max:    max,
	}
}

/*
StatArchDescriptiveVectorFiveNumberSummaryF64 computes the five-number summary in float64 precision.

The five-number summary consists of: Minimum, First Quartile (Q1), Median, Third Quartile (Q3), and Maximum.
This provides a comprehensive overview of the data distribution.

Use cases:
- Box plot construction
- Quick data distribution overview
- Outlier detection (using IQR = Q3 - Q1)
- Statistical reporting

Time complexity: O(n log n) - requires sorting the vector (first call only, then O(1) from cache)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector should contain at least 4 elements for meaningful quartiles
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns zero values if vector is empty
- For small vectors, quartiles may equal min/max/median

The function uses the analysis structure to efficiently compute and cache all five values.
All individual statistics (Min, Max, Median, Q1, Q3) are cached for reuse by other functions.
*/
func StatArchDescriptiveVectorFiveNumberSummaryF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) FiveNumberSummaryF64 {
	// Use cached values where available, compute others
	// Ensure min/max are cached by computing range if needed
	StatArchDescriptiveVectorRange(analysis)
	cachedMin, _ := core.GetCacheValue(analysis, core.StatKindMin)
	cachedMax, _ := core.GetCacheValue(analysis, core.StatKindMax)
	min := float64(cachedMin.(T))
	max := float64(cachedMax.(T))
	median := StatArchDescriptiveVectorMedianF64(analysis)
	q1 := StatArchDescriptiveVectorPercentileF64(analysis, 25)
	q3 := StatArchDescriptiveVectorPercentileF64(analysis, 75)

	return FiveNumberSummaryF64{
		Min:    min,
		Q1:     q1,
		Median: median,
		Q3:     q3,
		Max:    max,
	}
}

/*
StatArchDescriptiveVectorBimodalityCoefficientF32 computes the bimodality coefficient in float32 precision.

The bimodality coefficient measures whether a distribution has one or two modes (peaks).
Values greater than 0.555 suggest the distribution is bimodal (has two distinct peaks).

Use cases:
- Detecting bimodal distributions (two distinct populations)
- Data quality analysis (identifying mixed populations)
- Statistical modeling (determining if data should be split)
- Distribution shape analysis

Time complexity: O(1) - uses cached skewness and kurtosis values
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 2 elements (population) or 3 elements (sample) for skewness
- Vector must contain at least 3 elements (population) or 4 elements (sample) for kurtosis
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty
- Panics if vector has insufficient elements (inherited from skewness/kurtosis requirements)
- Formula: (skewness² + 1) / (kurtosis + 3)
- Range: 0 to 1 (values > 0.555 suggest bimodality)

The function uses the analysis structure to cache the computed bimodality coefficient.
It uses cached skewness and kurtosis values from the analysis structure.
*/
func StatArchDescriptiveVectorBimodalityCoefficientF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindBimodalityCoefficientF32); ok {
		return cached.(float32)
	}

	skewness := StatArchDescriptiveVectorSkewnessF32(analysis, sample)
	kurtosis := StatArchDescriptiveVectorKurtosisF32(analysis, sample)

	// Formula: (skewness² + 1) / (kurtosis + 3)
	// Note: kurtosis is already excess kurtosis (kurtosis - 3), so we add 3 back
	skewnessSquared := skewness * skewness
	bimodalityCoeff := (skewnessSquared + 1.0) / (kurtosis + 3.0)

	core.SetCacheValue(analysis, core.StatKindBimodalityCoefficientF32, bimodalityCoeff)
	return bimodalityCoeff
}

/*
StatArchDescriptiveVectorBimodalityCoefficientF64 computes the bimodality coefficient in float64 precision.

The bimodality coefficient measures whether a distribution has one or two modes (peaks).
Values greater than 0.555 suggest the distribution is bimodal (has two distinct peaks).

Use cases:
- Detecting bimodal distributions (two distinct populations)
- Data quality analysis (identifying mixed populations)
- Statistical modeling (determining if data should be split)
- Distribution shape analysis

Time complexity: O(1) - uses cached skewness and kurtosis values
Space complexity: O(1) - only local variables used

Prerequisites:
- Vector must contain at least 2 elements (population) or 3 elements (sample) for skewness
- Vector must contain at least 3 elements (population) or 4 elements (sample) for kurtosis
- Vector does not need to be sorted

Edge cases:
- Returns 0 if vector is empty
- Panics if vector has insufficient elements (inherited from skewness/kurtosis requirements)
- Formula: (skewness² + 1) / (kurtosis + 3)
- Range: 0 to 1 (values > 0.555 suggest bimodality)

The function uses the analysis structure to cache the computed bimodality coefficient.
It uses cached skewness and kurtosis values from the analysis structure.
*/
func StatArchDescriptiveVectorBimodalityCoefficientF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	if cached, ok := core.GetCacheValue(analysis, core.StatKindBimodalityCoefficientF64); ok {
		return cached.(float64)
	}

	skewness := StatArchDescriptiveVectorSkewnessF64(analysis, sample)
	kurtosis := StatArchDescriptiveVectorKurtosisF64(analysis, sample)

	// Formula: (skewness² + 1) / (kurtosis + 3)
	// Note: kurtosis is already excess kurtosis (kurtosis - 3), so we add 3 back
	skewnessSquared := skewness * skewness
	bimodalityCoeff := (skewnessSquared + 1.0) / (kurtosis + 3.0)

	core.SetCacheValue(analysis, core.StatKindBimodalityCoefficientF64, bimodalityCoeff)
	return bimodalityCoeff
}

/*
StatArchDescriptiveVectorQuantileBierensF32 computes a quantile using Bierens' kernel-based estimator in float32 precision.

Bierens' estimator uses kernel weights to provide a more robust quantile estimate than simple linear interpolation,
especially for small samples. It applies a kernel function to weight observations around the target quantile.

Use cases:
- Robust quantile estimation for small samples
- When simple percentile interpolation may be unreliable
- Statistical analysis requiring more stable quantile estimates

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- Percentile must be in range [0, 100]
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if percentile > 100
- Uses Epanechnikov kernel for weighting

Note: Bierens quantiles are not cached due to the large number of possible percentile values.
A sorted copy of the vector is created on first use and reused for subsequent operations.

Accuracy Note: Bandwidth selection uses a simplified adaptive approach. For research-grade accuracy,
consider more sophisticated bandwidth selection methods. See README "Accuracy and Simplifications" section for details.
*/
func StatArchDescriptiveVectorQuantileBierensF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	percentile uint8,
) float32 {
	if percentile > 100 {
		panic("percentile > 100")
	}

	sortedVector := core.GetSortedVector(analysis)
	length := analysis.Count
	if length == 0 {
		return 0
	}
	if length == 1 {
		return float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, 0))
	}

	// Target rank
	targetRank := float32(percentile) / 100.0 * float32(length-1)

	// Bandwidth (adaptive)
	bandwidth := float32(length)
	if bandwidth < 1.0 {
		bandwidth = 1.0
	}

	var weightedSum float32 = 0
	var weightSum float32 = 0

	// Epanechnikov kernel weights
	for i := uint64(0); i < length; i++ {
		rank := float32(i)
		u := (rank - targetRank) / bandwidth

		// Epanechnikov kernel: K(u) = 0.75 * (1 - u²) for |u| <= 1, else 0
		var weight float32 = 0
		if u*u <= 1.0 {
			weight = 0.75 * (1.0 - u*u)
		}

		if weight > 0 {
			val := float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, i))
			weightedSum += weight * val
			weightSum += weight
		}
	}

	if weightSum == 0 {
		// Fallback to simple percentile if no weights
		return StatArchDescriptiveVectorPercentileF32(analysis, percentile)
	}

	return weightedSum / weightSum
}

/*
StatArchDescriptiveVectorQuantileBierensF64 computes a quantile using Bierens' kernel-based estimator in float64 precision.

Bierens' estimator uses kernel weights to provide a more robust quantile estimate than simple linear interpolation,
especially for small samples. It applies a kernel function to weight observations around the target quantile.

Use cases:
- Robust quantile estimation for small samples
- When simple percentile interpolation may be unreliable
- Statistical analysis requiring more stable quantile estimates

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- Percentile must be in range [0, 100]
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if percentile > 100
- Uses Epanechnikov kernel for weighting

Note: Bierens quantiles are not cached due to the large number of possible percentile values.
A sorted copy of the vector is created on first use and reused for subsequent operations.

Accuracy Note: Bandwidth selection uses a simplified adaptive approach. For research-grade accuracy,
consider more sophisticated bandwidth selection methods. See README "Accuracy and Simplifications" section for details.
*/
func StatArchDescriptiveVectorQuantileBierensF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	percentile uint8,
) float64 {
	if percentile > 100 {
		panic("percentile > 100")
	}

	sortedVector := core.GetSortedVector(analysis)
	length := analysis.Count
	if length == 0 {
		return 0
	}
	if length == 1 {
		return float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, 0))
	}

	// Target rank
	targetRank := float64(percentile) / 100.0 * float64(length-1)

	// Bandwidth (adaptive)
	bandwidth := float64(length)
	if bandwidth < 1.0 {
		bandwidth = 1.0
	}

	var weightedSum float64 = 0
	var weightSum float64 = 0

	// Epanechnikov kernel weights
	for i := uint64(0); i < length; i++ {
		rank := float64(i)
		u := (rank - targetRank) / bandwidth

		// Epanechnikov kernel: K(u) = 0.75 * (1 - u²) for |u| <= 1, else 0
		var weight float64 = 0
		if u*u <= 1.0 {
			weight = 0.75 * (1.0 - u*u)
		}

		if weight > 0 {
			val := float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, i))
			weightedSum += weight * val
			weightSum += weight
		}
	}

	if weightSum == 0 {
		// Fallback to simple percentile if no weights
		return StatArchDescriptiveVectorPercentileF64(analysis, percentile)
	}

	return weightedSum / weightSum
}

/*
StatArchDescriptiveVectorQuantileHarrellDavisF32 computes a quantile using Harrell-Davis estimator in float32 precision.

The Harrell-Davis estimator uses beta distribution weights to provide a more robust quantile estimate.
It weights all observations based on their position relative to the target quantile using beta distribution probabilities.

Use cases:
- Robust quantile estimation, especially for small samples
- When percentile interpolation may be unreliable
- Statistical analysis requiring unbiased quantile estimates

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- Percentile must be in range [0, 100]
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if percentile > 100

Note: Harrell-Davis quantiles are not cached due to the large number of possible percentile values.
A sorted copy of the vector is created on first use and reused for subsequent operations.

Accuracy Note: This implementation uses the regularized incomplete beta function from blaze/math,
which provides high accuracy even in the tails (P99.9) for risk analysis and SLA monitoring.
*/
func StatArchDescriptiveVectorQuantileHarrellDavisF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	percentile uint8,
) float32 {
	if percentile > 100 {
		panic("percentile > 100")
	}

	sortedVector := core.GetSortedVector(analysis)
	length := analysis.Count
	if length == 0 {
		return 0
	}
	if length == 1 {
		return float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, 0))
	}

	p := float32(percentile) / 100.0
	n := float32(length)

	// Beta distribution parameters
	alpha := p * (n + 1.0)
	beta := (1.0 - p) * (n + 1.0)

	var weightedSum float32 = 0

	// Compute weights using proper beta distribution: weight_i = I_{x_i}(alpha, beta) - I_{x_{i-1}}(alpha, beta)
	// where x_i = (i+1)/(n+1) and x_0 = 0
	var prevBetaValue float32 = 0.0 // I_0(alpha, beta) = 0

	for i := uint64(0); i < length; i++ {
		// Normalized position: x_i = (i+1)/(n+1)
		j := float32(i) + 1.0
		x := j / (n + 1.0)

		// Compute I_{x_i}(alpha, beta) using regularized incomplete beta function
		currentBetaValue := math.BlazeMathBetaRegularizedIncompleteF32(x, alpha, beta)

		// Weight is the difference: I_{x_i} - I_{x_{i-1}}
		weight := currentBetaValue - prevBetaValue
		prevBetaValue = currentBetaValue

		val := float32(memstruct.VectorItemGetAtUnsafe[T](sortedVector, i))
		weightedSum += weight * val
	}

	return weightedSum
}

/*
StatArchDescriptiveVectorQuantileHarrellDavisF64 computes a quantile using Harrell-Davis estimator in float64 precision.

The Harrell-Davis estimator uses beta distribution weights to provide a more robust quantile estimate.
It weights all observations based on their position relative to the target quantile using beta distribution probabilities.

Use cases:
- Robust quantile estimation, especially for small samples
- When percentile interpolation may be unreliable
- Statistical analysis requiring unbiased quantile estimates

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- Percentile must be in range [0, 100]
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if percentile > 100

Note: Harrell-Davis quantiles are not cached due to the large number of possible percentile values.
A sorted copy of the vector is created on first use and reused for subsequent operations.

Accuracy Note: This implementation uses the regularized incomplete beta function from blaze/math,
which provides high accuracy even in the tails (P99.9) for risk analysis and SLA monitoring.
*/
func StatArchDescriptiveVectorQuantileHarrellDavisF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	percentile uint8,
) float64 {
	if percentile > 100 {
		panic("percentile > 100")
	}

	sortedVector := core.GetSortedVector(analysis)
	length := analysis.Count
	if length == 0 {
		return 0
	}
	if length == 1 {
		return float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, 0))
	}

	p := float64(percentile) / 100.0
	n := float64(length)

	// Beta distribution parameters
	alpha := p * (n + 1.0)
	beta := (1.0 - p) * (n + 1.0)

	var weightedSum float64 = 0

	// Compute weights using proper beta distribution: weight_i = I_{x_i}(alpha, beta) - I_{x_{i-1}}(alpha, beta)
	// where x_i = (i+1)/(n+1) and x_0 = 0
	var prevBetaValue float64 = 0.0 // I_0(alpha, beta) = 0

	for i := uint64(0); i < length; i++ {
		// Normalized position: x_i = (i+1)/(n+1)
		j := float64(i) + 1.0
		x := j / (n + 1.0)

		// Compute I_{x_i}(alpha, beta) using regularized incomplete beta function
		currentBetaValue := math.BlazeMathBetaRegularizedIncompleteF64(x, alpha, beta)

		// Weight is the difference: I_{x_i} - I_{x_{i-1}}
		weight := currentBetaValue - prevBetaValue
		prevBetaValue = currentBetaValue

		val := float64(memstruct.VectorItemGetAtUnsafe[T](sortedVector, i))
		weightedSum += weight * val
	}

	return weightedSum
}

/*
StatArchDescriptiveVectorQuantileRangeF32 computes the ratio of two quantiles in float32 precision.

The quantile range ratio measures the "long tail" intensity by comparing an upper quantile to a lower quantile.
Common examples: P99/P50 (99th percentile to median) or P95/P5 ratios.

Use cases:
- Measuring tail intensity in performance metrics (e.g., P99/P50 latency ratio)
- Economic analysis (income inequality, wealth distribution)
- Systems performance analysis (long tail detection)
- Distribution shape characterization

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- Both percentiles must be in range [0, 100]
- upperPercentile should be > lowerPercentile for meaningful results
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if either percentile > 100
- Panics if lowerPercentile is 0 (division by zero)
- Returns 1 if both percentiles are equal

Note: Quantile ratios are not cached due to the large number of possible percentile combinations.
*/
func StatArchDescriptiveVectorQuantileRangeF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	upperPercentile uint8,
	lowerPercentile uint8,
) float32 {
	if upperPercentile > 100 || lowerPercentile > 100 {
		panic("percentile > 100")
	}
	if lowerPercentile == 0 {
		panic("lowerPercentile cannot be 0 (division by zero)")
	}

	upperQuantile := StatArchDescriptiveVectorPercentileF32(analysis, upperPercentile)
	lowerQuantile := StatArchDescriptiveVectorPercentileF32(analysis, lowerPercentile)

	if lowerQuantile == 0 {
		panic("lower quantile is 0, cannot compute ratio")
	}

	return upperQuantile / lowerQuantile
}

/*
StatArchDescriptiveVectorQuantileRangeF64 computes the ratio of two quantiles in float64 precision.

The quantile range ratio measures the "long tail" intensity by comparing an upper quantile to a lower quantile.
Common examples: P99/P50 (99th percentile to median) or P95/P5 ratios.

Use cases:
- Measuring tail intensity in performance metrics (e.g., P99/P50 latency ratio)
- Economic analysis (income inequality, wealth distribution)
- Systems performance analysis (long tail detection)
- Distribution shape characterization

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- Both percentiles must be in range [0, 100]
- upperPercentile should be > lowerPercentile for meaningful results
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if either percentile > 100
- Panics if lowerPercentile is 0 (division by zero)
- Returns 1 if both percentiles are equal

Note: Quantile ratios are not cached due to the large number of possible percentile combinations.
*/
func StatArchDescriptiveVectorQuantileRangeF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	upperPercentile uint8,
	lowerPercentile uint8,
) float64 {
	if upperPercentile > 100 || lowerPercentile > 100 {
		panic("percentile > 100")
	}
	if lowerPercentile == 0 {
		panic("lowerPercentile cannot be 0 (division by zero)")
	}

	upperQuantile := StatArchDescriptiveVectorPercentileF64(analysis, upperPercentile)
	lowerQuantile := StatArchDescriptiveVectorPercentileF64(analysis, lowerPercentile)

	if lowerQuantile == 0 {
		panic("lower quantile is 0, cannot compute ratio")
	}

	return upperQuantile / lowerQuantile
}

/*
StatArchDescriptiveVectorDecileRatioF32 computes the decile ratio (P90/P10) in float32 precision.

The decile ratio is a specific quantile range ratio that compares the 90th percentile to the 10th percentile.
This is commonly used in economics and systems performance to measure inequality or tail intensity.

Use cases:
- Economic inequality measurement (income/wealth distribution)
- Systems performance analysis (P90/P10 latency ratio)
- Distribution tail intensity measurement
- Statistical reporting

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if P10 is 0 (division by zero)
- Returns 1 if P90 equals P10 (no spread)

Note: Decile ratio is not cached (computed from percentiles which are not cached).
*/
func StatArchDescriptiveVectorDecileRatioF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float32 {
	return StatArchDescriptiveVectorQuantileRangeF32(analysis, 90, 10)
}

/*
StatArchDescriptiveVectorDecileRatioF64 computes the decile ratio (P90/P10) in float64 precision.

The decile ratio is a specific quantile range ratio that compares the 90th percentile to the 10th percentile.
This is commonly used in economics and systems performance to measure inequality or tail intensity.

Use cases:
- Economic inequality measurement (income/wealth distribution)
- Systems performance analysis (P90/P10 latency ratio)
- Distribution tail intensity measurement
- Statistical reporting

Time complexity: O(n log n) - requires sorting the vector (first call only)
Space complexity: O(n) - creates sorted copy of vector on first call

Prerequisites:
- Vector must contain at least 1 element
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Returns 0 if vector is empty
- Panics if P10 is 0 (division by zero)
- Returns 1 if P90 equals P10 (no spread)

Note: Decile ratio is not cached (computed from percentiles which are not cached).
*/
func StatArchDescriptiveVectorDecileRatioF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
) float64 {
	return StatArchDescriptiveVectorQuantileRangeF64(analysis, 90, 10)
}

/*
StatArchDescriptiveVectorTopK extracts the top K largest values from a vector.

Top-K returns the K largest values in the vector, sorted in descending order.
This is useful when you need the actual extreme values, not just their statistics.

Use cases:
- Identifying top performers or outliers
- Systems performance analysis (top N slowest requests)
- Financial analysis (top N gains/losses)
- Data exploration (examining extreme values)

Time complexity: O(n log n) - requires sorting (first call only if sorted vector exists)
Space complexity: O(n) - uses scratch vector for sorting

Prerequisites:
- Vector must contain at least 1 element
- k must be > 0 and <= vector size
- destinationVector must have capacity >= k
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Panics if k is 0 or k > vector size
- Panics if destinationVector capacity < k
- If k equals vector size, returns all values sorted descending

Note: Top-K values are not cached (parameter-dependent, returns values not statistics).
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorTopK[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	k uint64,
	destinationVector memcore.MarkRaw,
) {
	size := analysis.Count
	if k == 0 || k > size {
		panic("k must be > 0 and <= vector size")
	}

	destCapacity := memstruct.VectorCapacityGet[T](destinationVector)
	if destCapacity < k {
		panic("destinationVector capacity is smaller than k")
	}

	sortedVector := core.GetSortedVector(analysis)

	// Copy top K values in descending order (from end of sorted vector)
	for i := uint64(0); i < k; i++ {
		srcIdx := size - 1 - i
		val := memstruct.VectorItemGetAtUnsafe[T](sortedVector, srcIdx)
		memstruct.VectorSetAtUnsafe(destinationVector, i, val)
	}
}

/*
StatArchDescriptiveVectorBottomK extracts the bottom K smallest values from a vector.

Bottom-K returns the K smallest values in the vector, sorted in ascending order.
This is useful when you need the actual extreme values, not just their statistics.

Use cases:
- Identifying bottom performers or outliers
- Systems performance analysis (bottom N fastest requests)
- Financial analysis (bottom N gains/losses)
- Data exploration (examining extreme values)

Time complexity: O(n log n) - requires sorting (first call only if sorted vector exists)
Space complexity: O(n) - uses scratch vector for sorting

Prerequisites:
- Vector must contain at least 1 element
- k must be > 0 and <= vector size
- destinationVector must have capacity >= k
- Vector does not need to be pre-sorted (function handles sorting internally)

Edge cases:
- Panics if k is 0 or k > vector size
- Panics if destinationVector capacity < k
- If k equals vector size, returns all values sorted ascending

Note: Bottom-K values are not cached (parameter-dependent, returns values not statistics).
A sorted copy of the vector is created on first use and reused for subsequent operations.
*/
func StatArchDescriptiveVectorBottomK[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	k uint64,
	destinationVector memcore.MarkRaw,
) {
	size := analysis.Count
	if k == 0 || k > size {
		panic("k must be > 0 and <= vector size")
	}

	destCapacity := memstruct.VectorCapacityGet[T](destinationVector)
	if destCapacity < k {
		panic("destinationVector capacity is smaller than k")
	}

	sortedVector := core.GetSortedVector(analysis)

	// Copy bottom K values in ascending order (from start of sorted vector)
	for i := uint64(0); i < k; i++ {
		val := memstruct.VectorItemGetAtUnsafe[T](sortedVector, i)
		memstruct.VectorSetAtUnsafe(destinationVector, i, val)
	}
}

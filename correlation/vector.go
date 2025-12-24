package correlation

import (
	"blaze/reduce"
	"foundation"
	"memarch"
	"memcore"
	"memstruct"
	"statarch/core"
	"statarch/descriptive"
)

/*
StatArchCorrelationVectorCovarianceF32 computes the covariance between two vectors in float32 precision.

Covariance measures the joint variability of two variables. Positive covariance indicates
that variables tend to increase together, negative indicates inverse relationship.

Use cases:
- Measuring linear relationship strength and direction
- Feature analysis in machine learning
- Financial analysis (portfolio risk, asset relationships)
- Quality control (process variable relationships)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 2 elements
- If sample=true, vectors represent samples from larger populations
- If sample=false, vectors represent entire populations
- Vectors do not need to be sorted

Edge cases:
- Panics if vectors have different lengths
- Panics if either vector has < 2 elements
- Returns 0 if variables are independent (zero covariance)
- Sample covariance uses Bessel's correction (divides by n-1 instead of n)

Formula: cov(X,Y) = E[(X - μX)(Y - μY)] with sample/population correction
*/
func StatArchCorrelationVectorCovarianceF32[T foundation.Numeric](
	analysisA *core.StatArchAnalysis[T],
	analysisB *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysisA)
	core.StatArchAnalysisValidateVersion(analysisB)

	sizeA := analysisA.Count
	sizeB := analysisB.Count

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA < 2 {
		panic("cannot compute covariance with less than 2 elements")
	}

	// Use cached means if available, otherwise compute and cache
	meanA := descriptive.StatArchDescriptiveVectorMeanF32(analysisA)
	meanB := descriptive.StatArchDescriptiveVectorMeanF32(analysisB)

	// Compute covariance using Blaze
	covariance := reduce.BlazeReduceCovarianceF32[T, T](analysisA.Vector, analysisB.Vector, meanA, meanB)

	denominator := float32(sizeA)
	if sample {
		denominator = float32(sizeA) - 1
	}

	return covariance / denominator
}

/*
StatArchCorrelationVectorCovarianceF64 computes the covariance between two vectors in float64 precision.

Covariance measures the joint variability of two variables. Positive covariance indicates
that variables tend to increase together, negative indicates inverse relationship.

Use cases:
- Measuring linear relationship strength and direction
- Feature analysis in machine learning
- Financial analysis (portfolio risk, asset relationships)
- Quality control (process variable relationships)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 2 elements
- If sample=true, vectors represent samples from larger populations
- If sample=false, vectors represent entire populations
- Vectors do not need to be sorted

Edge cases:
- Panics if vectors have different lengths
- Panics if either vector has < 2 elements
- Returns 0 if variables are independent (zero covariance)
- Sample covariance uses Bessel's correction (divides by n-1 instead of n)

Formula: cov(X,Y) = E[(X - μX)(Y - μY)] with sample/population correction
*/
func StatArchCorrelationVectorCovarianceF64[T foundation.Numeric](
	analysisA *core.StatArchAnalysis[T],
	analysisB *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysisA)
	core.StatArchAnalysisValidateVersion(analysisB)

	sizeA := analysisA.Count
	sizeB := analysisB.Count

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA < 2 {
		panic("cannot compute covariance with less than 2 elements")
	}

	// Use cached means if available, otherwise compute and cache
	meanA := descriptive.StatArchDescriptiveVectorMeanF64(analysisA)
	meanB := descriptive.StatArchDescriptiveVectorMeanF64(analysisB)

	// Compute covariance using Blaze
	covariance := reduce.BlazeReduceCovarianceF64[T, T](analysisA.Vector, analysisB.Vector, meanA, meanB)

	denominator := float64(sizeA)
	if sample {
		denominator = float64(sizeA) - 1
	}

	return covariance / denominator
}

/*
StatArchCorrelationVectorPearsonF32 computes the Pearson correlation coefficient in float32 precision.

Pearson correlation measures the linear relationship between two variables.
It is the normalized covariance, ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation).

Use cases:
- Measuring linear relationship strength and direction
- Feature selection in machine learning
- Financial analysis (asset correlation, portfolio diversification)
- Quality control (process variable relationships)

Time complexity: O(n) - depends on covariance and standard deviation calculations
Space complexity: O(1) - only local variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 2 elements
- If sample=true, vectors represent samples from larger populations
- If sample=false, vectors represent entire populations
- Vectors do not need to be sorted

Edge cases:
- Panics if vectors have different lengths
- Panics if either vector has < 2 elements
- Panics if either standard deviation is 0 (division by zero)
- Returns 0 if variables are uncorrelated
- Range: -1 to +1

Formula: r = cov(X,Y) / (σX * σY)
*/
func StatArchCorrelationVectorPearsonF32[T foundation.Numeric](
	analysisA *core.StatArchAnalysis[T],
	analysisB *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysisA)
	core.StatArchAnalysisValidateVersion(analysisB)

	sizeA := analysisA.Count
	sizeB := analysisB.Count

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA < 2 {
		panic("cannot compute Pearson correlation with less than 2 elements")
	}

	// Use cached means if available
	meanA := descriptive.StatArchDescriptiveVectorMeanF32(analysisA)
	meanB := descriptive.StatArchDescriptiveVectorMeanF32(analysisB)

	// Compute covariance using Blaze (reusing cached means)
	covarianceSum := reduce.BlazeReduceCovarianceF32[T, T](analysisA.Vector, analysisB.Vector, meanA, meanB)
	denominator := float32(sizeA)
	if sample {
		denominator = float32(sizeA) - 1
	}
	covariance := covarianceSum / denominator

	// Use cached standard deviations if available
	stddevA := descriptive.StatArchDescriptiveVectorStandardDeviationF32(analysisA, sample)
	stddevB := descriptive.StatArchDescriptiveVectorStandardDeviationF32(analysisB, sample)

	if stddevA == 0 || stddevB == 0 {
		panic("standard deviation is 0, cannot compute correlation")
	}

	return covariance / (stddevA * stddevB)
}

/*
StatArchCorrelationVectorPearsonF64 computes the Pearson correlation coefficient in float64 precision.

Pearson correlation measures the linear relationship between two variables.
It is the normalized covariance, ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation).

This function accepts StatArchAnalysis structures and uses cached means and standard deviations
if available, avoiding redundant computations. The correlation result itself is not cached since
it's computed between two vectors.

Use cases:
- Measuring linear relationship strength and direction
- Feature selection in machine learning
- Financial analysis (asset correlation, portfolio diversification)
- Quality control (process variable relationships)

Time complexity: O(n) - depends on covariance and standard deviation calculations
Space complexity: O(1) - only local variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 2 elements
- If sample=true, vectors represent samples from larger populations
- If sample=false, vectors represent entire populations
- Vectors do not need to be sorted

Edge cases:
- Panics if vectors have different lengths
- Panics if either vector has < 2 elements
- Panics if either standard deviation is 0 (division by zero)
- Returns 0 if variables are uncorrelated
- Range: -1 to +1

Formula: r = cov(X,Y) / (σX * σY)
*/
func StatArchCorrelationVectorPearsonF64[T foundation.Numeric](
	analysisA *core.StatArchAnalysis[T],
	analysisB *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysisA)
	core.StatArchAnalysisValidateVersion(analysisB)

	sizeA := analysisA.Count
	sizeB := analysisB.Count

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA < 2 {
		panic("cannot compute Pearson correlation with less than 2 elements")
	}

	// Use cached means if available
	meanA := descriptive.StatArchDescriptiveVectorMeanF64(analysisA)
	meanB := descriptive.StatArchDescriptiveVectorMeanF64(analysisB)

	// Compute covariance using Blaze (reusing cached means)
	covarianceSum := reduce.BlazeReduceCovarianceF64[T, T](analysisA.Vector, analysisB.Vector, meanA, meanB)
	denominator := float64(sizeA)
	if sample {
		denominator = float64(sizeA) - 1
	}
	covariance := covarianceSum / denominator

	// Use cached standard deviations if available
	stddevA := descriptive.StatArchDescriptiveVectorStandardDeviationF64(analysisA, sample)
	stddevB := descriptive.StatArchDescriptiveVectorStandardDeviationF64(analysisB, sample)

	if stddevA == 0 || stddevB == 0 {
		panic("standard deviation is 0, cannot compute correlation")
	}

	return covariance / (stddevA * stddevB)
}

/*
StatArchCorrelationVectorSpearmanF32 computes the Spearman rank correlation coefficient in float32 precision.

Spearman correlation measures the monotonic relationship between two variables using their ranks.
It is more robust to outliers than Pearson correlation and detects non-linear monotonic relationships.

Use cases:
- Measuring monotonic relationships (not just linear)
- Robust correlation when data has outliers
- Non-parametric analysis
- Quality control (rank-based process relationships)

Time complexity: O(n log n) - requires ranking both vectors (sorting)
Space complexity: O(n) - requires scratch vectors for ranking

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 2 elements
- allocFn must be provided for creating scratch vectors
- Vectors do not need to be sorted

Edge cases:
- Panics if vectors have different lengths
- Panics if either vector has < 2 elements
- Handles ties by assigning average ranks
- Range: -1 to +1

Implementation: Ranks both vectors using a single Array of rankEntry structs that bind values
with their original indices. This approach sorts once using O(n log n) quicksort, then assigns
ranks linearly. Ranks both vectors, then computes Pearson correlation on ranks.
*/
func StatArchCorrelationVectorSpearmanF32[T foundation.Numeric](
	vectorA memcore.MarkRaw,
	vectorB memcore.MarkRaw,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) float32 {
	sizeA := memstruct.VectorCapacityGet[T](vectorA)
	sizeB := memstruct.VectorCapacityGet[T](vectorB)

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA < 2 {
		panic("cannot compute Spearman correlation with less than 2 elements")
	}

	// Create scratch vectors for ranks
	rankVectorA, _ := memarch.MemArchVectorCreate[float32](allocFn, sizeA)
	rankVectorB, _ := memarch.MemArchVectorCreate[float32](allocFn, sizeB)

	// Rank vector A
	rankVectorF32[T](rankVectorA, vectorA, sizeA, allocFn)

	// Rank vector B
	rankVectorF32[T](rankVectorB, vectorB, sizeB, allocFn)

	// Create analysis structures for rank vectors
	rankAnalysisA := core.StatArchAnalysisCreate[float32](rankVectorA, allocFn)
	rankAnalysisB := core.StatArchAnalysisCreate[float32](rankVectorB, allocFn)

	// Compute Pearson correlation on ranks
	return StatArchCorrelationVectorPearsonF32(rankAnalysisA, rankAnalysisB, true)
}

/*
StatArchCorrelationVectorSpearmanF64 computes the Spearman rank correlation coefficient in float64 precision.

Spearman correlation measures the monotonic relationship between two variables using their ranks.
It is more robust to outliers than Pearson correlation and detects non-linear monotonic relationships.

Use cases:
- Measuring monotonic relationships (not just linear)
- Robust correlation when data has outliers
- Non-parametric analysis
- Quality control (rank-based process relationships)

Time complexity: O(n log n) - requires ranking both vectors (sorting)
Space complexity: O(n) - requires scratch vectors for ranking

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 2 elements
- allocFn must be provided for creating scratch vectors
- Vectors do not need to be sorted

Edge cases:
- Panics if vectors have different lengths
- Panics if either vector has < 2 elements
- Handles ties by assigning average ranks
- Range: -1 to +1

Implementation: Ranks both vectors using a single Array of rankEntryF64 structs that bind values
with their original indices. This approach sorts once using O(n log n) quicksort, then assigns
ranks linearly. Ranks both vectors, then computes Pearson correlation on ranks.
*/
func StatArchCorrelationVectorSpearmanF64[T foundation.Numeric](
	vectorA memcore.MarkRaw,
	vectorB memcore.MarkRaw,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) float64 {
	sizeA := memstruct.VectorCapacityGet[T](vectorA)
	sizeB := memstruct.VectorCapacityGet[T](vectorB)

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA < 2 {
		panic("cannot compute Spearman correlation with less than 2 elements")
	}

	// Create scratch vectors for ranks
	rankVectorA, _ := memarch.MemArchVectorCreate[float64](allocFn, sizeA)
	rankVectorB, _ := memarch.MemArchVectorCreate[float64](allocFn, sizeB)

	// Rank vector A
	rankVectorF64[T](rankVectorA, vectorA, sizeA, allocFn)

	// Rank vector B
	rankVectorF64[T](rankVectorB, vectorB, sizeB, allocFn)

	// Create analysis structures for rank vectors
	rankAnalysisA := core.StatArchAnalysisCreate[float64](rankVectorA, allocFn)
	rankAnalysisB := core.StatArchAnalysisCreate[float64](rankVectorB, allocFn)

	// Compute Pearson correlation on ranks
	return StatArchCorrelationVectorPearsonF64(rankAnalysisA, rankAnalysisB, true)
}

// rankEntry binds a value with its original index for ranking purposes.
type rankEntry[T foundation.Numeric] struct {
	Value float32
	Index uint64
}

// rankVectorF32 computes ranks for a vector in float32, handling ties with average ranks.
//
// Uses a single Array of rankEntry structs that bind values with their original indices.
// This approach sorts once and then assigns ranks linearly, providing O(n log n) performance.
func rankVectorF32[T foundation.Numeric](
	rankVector memcore.MarkRaw,
	sourceVector memcore.MarkRaw,
	size uint64,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) {
	// Create Array of rankEntry structs
	entries, _ := memarch.MemArchArrayCreate[rankEntry[T]](allocFn, size)

	// Fill: Pass through the source vector once, filling entries with value and index
	var idx uint64 = 0
	memstruct.VectorUnaryReadOnlyExecute(sourceVector, func(item T) {
		entry := rankEntry[T]{
			Value: float32(item),
			Index: idx,
		}
		memstruct.ArraySetAtUnsafe(entries, idx, entry)
		idx++
	}, 1)

	// Sort: Call ArraySort on the entry array. The comparator only looks at Entry.Value.
	memstruct.ArraySort(entries, func(a, b rankEntry[T]) int {
		if a.Value < b.Value {
			return -1
		}
		if a.Value > b.Value {
			return 1
		}
		return 0
	})

	// Assign: Iterate through the sorted entry array. Because it's sorted, ties (equal values)
	// will be contiguous. Calculate the average rank for a block of tied entries and write
	// that rank back to the rankVector at the position specified by Entry.Index.
	var currentRank float32 = 1
	for i := uint64(0); i < size; {
		// Count how many values are equal (ties)
		tieCount := uint64(1)
		currentEntry := memstruct.ArrayItemGetAtUnsafe[rankEntry[T]](entries, i)
		currentVal := currentEntry.Value

		for j := i + 1; j < size; j++ {
			nextEntry := memstruct.ArrayItemGetAtUnsafe[rankEntry[T]](entries, j)
			if nextEntry.Value == currentVal {
				tieCount++
			} else {
				break
			}
		}

		// Average rank for ties
		avgRank := currentRank + float32(tieCount-1)/2.0

		// Assign rank to all tied values at their original positions
		for j := uint64(0); j < tieCount; j++ {
			entry := memstruct.ArrayItemGetAtUnsafe[rankEntry[T]](entries, i+j)
			memstruct.VectorSetAtUnsafe(rankVector, entry.Index, avgRank)
		}

		i += tieCount
		currentRank += float32(tieCount)
	}
}

// rankEntryF64 binds a value with its original index for ranking purposes in float64 precision.
type rankEntryF64[T foundation.Numeric] struct {
	Value float64
	Index uint64
}

// rankVectorF64 computes ranks for a vector in float64, handling ties with average ranks.
//
// Uses a single Array of rankEntryF64 structs that bind values with their original indices.
// This approach sorts once and then assigns ranks linearly, providing O(n log n) performance.
func rankVectorF64[T foundation.Numeric](
	rankVector memcore.MarkRaw,
	sourceVector memcore.MarkRaw,
	size uint64,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) {
	// Create Array of rankEntryF64 structs
	entries, _ := memarch.MemArchArrayCreate[rankEntryF64[T]](allocFn, size)

	// Fill: Pass through the source vector once, filling entries with value and index
	var idx uint64 = 0
	memstruct.VectorUnaryReadOnlyExecute(sourceVector, func(item T) {
		entry := rankEntryF64[T]{
			Value: float64(item),
			Index: idx,
		}
		memstruct.ArraySetAtUnsafe(entries, idx, entry)
		idx++
	}, 1)

	// Sort: Call ArraySort on the entry array. The comparator only looks at Entry.Value.
	memstruct.ArraySort(entries, func(a, b rankEntryF64[T]) int {
		if a.Value < b.Value {
			return -1
		}
		if a.Value > b.Value {
			return 1
		}
		return 0
	})

	// Assign: Iterate through the sorted entry array. Because it's sorted, ties (equal values)
	// will be contiguous. Calculate the average rank for a block of tied entries and write
	// that rank back to the rankVector at the position specified by Entry.Index.
	var currentRank float64 = 1
	for i := uint64(0); i < size; {
		// Count how many values are equal (ties)
		tieCount := uint64(1)
		currentEntry := memstruct.ArrayItemGetAtUnsafe[rankEntryF64[T]](entries, i)
		currentVal := currentEntry.Value

		for j := i + 1; j < size; j++ {
			nextEntry := memstruct.ArrayItemGetAtUnsafe[rankEntryF64[T]](entries, j)
			if nextEntry.Value == currentVal {
				tieCount++
			} else {
				break
			}
		}

		// Average rank for ties
		avgRank := currentRank + float64(tieCount-1)/2.0

		// Assign rank to all tied values at their original positions
		for j := uint64(0); j < tieCount; j++ {
			entry := memstruct.ArrayItemGetAtUnsafe[rankEntryF64[T]](entries, i+j)
			memstruct.VectorSetAtUnsafe(rankVector, entry.Index, avgRank)
		}

		i += tieCount
		currentRank += float64(tieCount)
	}
}

/*
StatArchCorrelationVectorCosineSimilarityF32 computes the cosine similarity between two vectors in float32 precision.

Cosine similarity measures the cosine of the angle between two vectors, indicating their directional similarity.
It is commonly used for high-dimensional data and text analysis.

Use cases:
- Text similarity (document vectors, word embeddings)
- Recommendation systems (user/item similarity)
- Machine learning (feature similarity)
- Information retrieval

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 1 element
- Vectors do not need to be sorted

Edge cases:
- Panics if vectors have different lengths
- Panics if either vector has zero norm (division by zero)
- Range: -1 to +1 (typically 0 to 1 for non-negative vectors)

Formula: cos(θ) = (A · B) / (||A|| * ||B||)
*/
func StatArchCorrelationVectorCosineSimilarityF32[T foundation.Numeric](
	vectorA memcore.MarkRaw,
	vectorB memcore.MarkRaw,
) float32 {
	sizeA := memstruct.VectorCapacityGet[T](vectorA)
	sizeB := memstruct.VectorCapacityGet[T](vectorB)

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA == 0 {
		panic("cannot compute cosine similarity with empty vectors")
	}

	// Compute dot product and norms using Blaze
	dotProduct := reduce.BlazeReduceDotProductF32[T, T](vectorA, vectorB)
	normASq := reduce.BlazeReduceVectorSumSquaredF32[T](vectorA)
	normBSq := reduce.BlazeReduceVectorSumSquaredF32[T](vectorB)

	normA := foundation.Sqrt32(normASq)
	normB := foundation.Sqrt32(normBSq)

	if normA == 0 || normB == 0 {
		panic("vector norm is 0, cannot compute cosine similarity")
	}

	return dotProduct / (normA * normB)
}

/*
StatArchCorrelationVectorCosineSimilarityF64 computes the cosine similarity between two vectors in float64 precision.

Cosine similarity measures the cosine of the angle between two vectors, indicating their directional similarity.
It is commonly used for high-dimensional data and text analysis.

Use cases:
- Text similarity (document vectors, word embeddings)
- Recommendation systems (user/item similarity)
- Machine learning (feature similarity)
- Information retrieval

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 1 element
- Vectors do not need to be sorted

Edge cases:
- Panics if vectors have different lengths
- Panics if either vector has zero norm (division by zero)
- Range: -1 to +1 (typically 0 to 1 for non-negative vectors)

Formula: cos(θ) = (A · B) / (||A|| * ||B||)
*/
func StatArchCorrelationVectorCosineSimilarityF64[T foundation.Numeric](
	vectorA memcore.MarkRaw,
	vectorB memcore.MarkRaw,
) float64 {
	sizeA := memstruct.VectorCapacityGet[T](vectorA)
	sizeB := memstruct.VectorCapacityGet[T](vectorB)

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA == 0 {
		panic("cannot compute cosine similarity with empty vectors")
	}

	// Compute dot product and norms using Blaze
	dotProduct := reduce.BlazeReduceDotProductF64[T, T](vectorA, vectorB)
	normASq := reduce.BlazeReduceVectorSumSquaredF64[T](vectorA)
	normBSq := reduce.BlazeReduceVectorSumSquaredF64[T](vectorB)

	normA := foundation.Sqrt64(normASq)
	normB := foundation.Sqrt64(normBSq)

	if normA == 0 || normB == 0 {
		panic("vector norm is 0, cannot compute cosine similarity")
	}

	return dotProduct / (normA * normB)
}

/*
StatArchCorrelationVectorAutocorrelationF32 computes the autocorrelation at a given lag in float32 precision.

Autocorrelation (also known as serial correlation) measures the correlation of a signal with a delayed copy of itself.
It is commonly used in time series analysis to detect patterns, periodicity, or randomness.

Use cases:
- Time series analysis (detecting trends, cycles, seasonality)
- Signal processing (detecting periodic patterns)
- Financial analysis (detecting momentum or mean reversion)
- Quality control (detecting process drift or cycles)
- Detecting randomness (random data should have autocorrelation near 0 at all lags)

Time complexity: O(n) - single pass through vector portions
Space complexity: O(n) - requires temporary vectors for original and lagged portions

Prerequisites:
- Vector must contain at least 2 elements
- Lag must be >= 0 and < vector length
- After lag, there must be at least 2 elements remaining for correlation computation
- allocFn must be provided for creating temporary vectors
- Vector does not need to be sorted

Edge cases:
- Panics if vector has < 2 elements
- Panics if lag >= vector length
- Panics if (vector length - lag) < 2 (not enough data for correlation)
- Returns 1.0 at lag 0 (perfect correlation with itself)
- Range: -1 to +1

Formula: r(k) = corr(X[0:n-k], X[k:n]) where k is the lag
*/
func StatArchCorrelationVectorAutocorrelationF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	lag uint64,
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	size := analysis.Count

	if size < 2 {
		panic("cannot compute autocorrelation with less than 2 elements")
	}
	if lag >= size {
		panic("lag must be less than vector length")
	}

	effectiveSize := size - lag
	if effectiveSize < 2 {
		panic("not enough elements remaining after lag for correlation computation")
	}

	if lag == 0 {
		return 1.0
	}

	// Create temporary vectors for original and lagged portions
	originalVector, _ := memarch.MemArchVectorCreate[T](analysis.AllocFn, effectiveSize)
	laggedVector, _ := memarch.MemArchVectorCreate[T](analysis.AllocFn, effectiveSize)

	// Copy original portion: X[0] to X[effectiveSize-1]
	for i := uint64(0); i < effectiveSize; i++ {
		val := memstruct.VectorItemGetAtUnsafe[T](analysis.Vector, i)
		memstruct.VectorSetAtUnsafe(originalVector, i, val)
	}

	// Copy lagged portion: X[lag] to X[lag+effectiveSize-1]
	for i := uint64(0); i < effectiveSize; i++ {
		val := memstruct.VectorItemGetAtUnsafe[T](analysis.Vector, lag+i)
		memstruct.VectorSetAtUnsafe(laggedVector, i, val)
	}

	// Create analysis structures for both portions
	originalAnalysis := core.StatArchAnalysisCreate[T](originalVector, analysis.AllocFn)
	laggedAnalysis := core.StatArchAnalysisCreate[T](laggedVector, analysis.AllocFn)

	// Compute Pearson correlation between original and lagged portions
	return StatArchCorrelationVectorPearsonF32(originalAnalysis, laggedAnalysis, sample)
}

/*
StatArchCorrelationVectorAutocorrelationF64 computes the autocorrelation at a given lag in float64 precision.

Autocorrelation (also known as serial correlation) measures the correlation of a signal with a delayed copy of itself.
It is commonly used in time series analysis to detect patterns, periodicity, or randomness.

Use cases:
- Time series analysis (detecting trends, cycles, seasonality)
- Signal processing (detecting periodic patterns)
- Financial analysis (detecting momentum or mean reversion)
- Quality control (detecting process drift or cycles)
- Detecting randomness (random data should have autocorrelation near 0 at all lags)

Time complexity: O(n) - single pass through vector portions
Space complexity: O(n) - requires temporary vectors for original and lagged portions

Prerequisites:
- Vector must contain at least 2 elements
- Lag must be >= 0 and < vector length
- After lag, there must be at least 2 elements remaining for correlation computation
- allocFn must be provided for creating temporary vectors
- Vector does not need to be sorted

Edge cases:
- Panics if vector has < 2 elements
- Panics if lag >= vector length
- Panics if (vector length - lag) < 2 (not enough data for correlation)
- Returns 1.0 at lag 0 (perfect correlation with itself)
- Range: -1 to +1

Formula: r(k) = corr(X[0:n-k], X[k:n]) where k is the lag
*/
func StatArchCorrelationVectorAutocorrelationF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	lag uint64,
	sample bool,
) float64 {
	size := analysis.Count

	if size < 2 {
		panic("cannot compute autocorrelation with less than 2 elements")
	}
	if lag >= size {
		panic("lag must be less than vector length")
	}

	effectiveSize := size - lag
	if effectiveSize < 2 {
		panic("not enough elements remaining after lag for correlation computation")
	}

	if lag == 0 {
		return 1.0
	}

	// Create temporary vectors for original and lagged portions
	originalVector, _ := memarch.MemArchVectorCreate[T](analysis.AllocFn, effectiveSize)
	laggedVector, _ := memarch.MemArchVectorCreate[T](analysis.AllocFn, effectiveSize)

	// Copy original portion: X[0] to X[effectiveSize-1]
	for i := uint64(0); i < effectiveSize; i++ {
		val := memstruct.VectorItemGetAtUnsafe[T](analysis.Vector, i)
		memstruct.VectorSetAtUnsafe(originalVector, i, val)
	}

	// Copy lagged portion: X[lag] to X[lag+effectiveSize-1]
	for i := uint64(0); i < effectiveSize; i++ {
		val := memstruct.VectorItemGetAtUnsafe[T](analysis.Vector, lag+i)
		memstruct.VectorSetAtUnsafe(laggedVector, i, val)
	}

	// Create analysis structures for both portions
	originalAnalysis := core.StatArchAnalysisCreate[T](originalVector, analysis.AllocFn)
	laggedAnalysis := core.StatArchAnalysisCreate[T](laggedVector, analysis.AllocFn)

	// Compute Pearson correlation between original and lagged portions
	return StatArchCorrelationVectorPearsonF64(originalAnalysis, laggedAnalysis, sample)
}

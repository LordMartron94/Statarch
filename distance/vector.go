package distance

import (
	"foundation"
	"memstruct"
	"statarch/core"
	"statarch/descriptive"
)

/*
StatArchDistanceVectorBhattacharyyaF32 computes the Bhattacharyya distance between two probability distributions in float32 precision.

Bhattacharyya distance measures the similarity between two probability distributions.
It is based on the Bhattacharyya coefficient, which measures the amount of overlap between distributions.

Use cases:
- Detecting distribution drift (model performance monitoring)
- Comparing probability distributions
- Measuring distribution similarity
- Statistical hypothesis testing

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 1 element
- Vectors are treated as probability distributions (will be normalized)
- All values must be non-negative (probabilities)

Edge cases:
- Panics if vectors have different lengths
- Panics if any value is negative
- Returns 0 if distributions are identical
- Range: 0 to ∞ (0 = identical distributions)

Formula: D_B = -ln(Σ√(p_i * q_i)) where p, q are normalized probability vectors
*/
func StatArchDistanceVectorBhattacharyyaF32[T foundation.Numeric](
	analysisA *core.StatArchAnalysis[T],
	analysisB *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysisA)
	core.StatArchAnalysisValidateVersion(analysisB)

	sizeA := analysisA.Count
	sizeB := analysisB.Count

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA == 0 {
		panic("cannot compute Bhattacharyya distance with empty vectors")
	}

	// Check for negative values
	memstruct.VectorUnaryReadOnlyExecute(analysisA.Vector, func(item T) {
		val := float32(item)
		if val < 0 {
			panic("Bhattacharyya distance requires non-negative values")
		}
	}, core.StatarchDefaultStride)
	memstruct.VectorUnaryReadOnlyExecute(analysisB.Vector, func(item T) {
		val := float32(item)
		if val < 0 {
			panic("Bhattacharyya distance requires non-negative values")
		}
	}, core.StatarchDefaultStride)

	// Use cached sums if available, otherwise compute and cache
	sumA := descriptive.StatArchDescriptiveVectorSumF32(analysisA)
	sumB := descriptive.StatArchDescriptiveVectorSumF32(analysisB)

	if sumA == 0 || sumB == 0 {
		panic("cannot compute Bhattacharyya distance: vector sum is 0")
	}

	// Compute Bhattacharyya coefficient
	var bhattacharyyaCoeff float32 = 0
	memstruct.VectorBinaryReadOnlyExecute(analysisA.Vector, analysisB.Vector, func(itemA T, itemB T) {
		p := float32(itemA) / sumA
		q := float32(itemB) / sumB
		bhattacharyyaCoeff += foundation.Sqrt32(p * q)
	}, core.StatarchDefaultStride)

	if bhattacharyyaCoeff <= 0 {
		panic("Bhattacharyya coefficient is non-positive")
	}

	// Distance is negative log of coefficient
	return -foundation.Log32(bhattacharyyaCoeff)
}

/*
StatArchDistanceVectorBhattacharyyaF64 computes the Bhattacharyya distance between two probability distributions in float64 precision.

Bhattacharyya distance measures the similarity between two probability distributions.
It is based on the Bhattacharyya coefficient, which measures the amount of overlap between distributions.

Use cases:
- Detecting distribution drift (model performance monitoring)
- Comparing probability distributions
- Measuring distribution similarity
- Statistical hypothesis testing

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 1 element
- Vectors are treated as probability distributions (will be normalized)
- All values must be non-negative (probabilities)

Edge cases:
- Panics if vectors have different lengths
- Panics if any value is negative
- Returns 0 if distributions are identical
- Range: 0 to ∞ (0 = identical distributions)

Formula: D_B = -ln(Σ√(p_i * q_i)) where p, q are normalized probability vectors
*/
func StatArchDistanceVectorBhattacharyyaF64[T foundation.Numeric](
	analysisA *core.StatArchAnalysis[T],
	analysisB *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysisA)
	core.StatArchAnalysisValidateVersion(analysisB)

	sizeA := analysisA.Count
	sizeB := analysisB.Count

	if sizeA != sizeB {
		panic("vectors must have the same length")
	}
	if sizeA == 0 {
		panic("cannot compute Bhattacharyya distance with empty vectors")
	}

	// Check for negative values
	memstruct.VectorUnaryReadOnlyExecute(analysisA.Vector, func(item T) {
		val := float64(item)
		if val < 0 {
			panic("Bhattacharyya distance requires non-negative values")
		}
	}, core.StatarchDefaultStride)
	memstruct.VectorUnaryReadOnlyExecute(analysisB.Vector, func(item T) {
		val := float64(item)
		if val < 0 {
			panic("Bhattacharyya distance requires non-negative values")
		}
	}, core.StatarchDefaultStride)

	// Use cached sums if available, otherwise compute and cache
	sumA := descriptive.StatArchDescriptiveVectorSumF64(analysisA)
	sumB := descriptive.StatArchDescriptiveVectorSumF64(analysisB)

	if sumA == 0 || sumB == 0 {
		panic("cannot compute Bhattacharyya distance: vector sum is 0")
	}

	// Compute Bhattacharyya coefficient
	var bhattacharyyaCoeff float64 = 0
	memstruct.VectorBinaryReadOnlyExecute(analysisA.Vector, analysisB.Vector, func(itemA T, itemB T) {
		p := float64(itemA) / sumA
		q := float64(itemB) / sumB
		bhattacharyyaCoeff += foundation.Sqrt64(p * q)
	}, core.StatarchDefaultStride)

	if bhattacharyyaCoeff <= 0 {
		panic("Bhattacharyya coefficient is non-positive")
	}

	// Distance is negative log of coefficient
	return -foundation.Log64(bhattacharyyaCoeff)
}

/*
StatArchDistanceVectorKLDivergenceF32 computes the Kullback-Leibler divergence in float32 precision.

KL divergence measures how much one probability distribution diverges from another.
It is asymmetric: KL(P||Q) ≠ KL(Q||P), where P is the "true" distribution and Q is the approximation.

Use cases:
- Detecting model drift (comparing training vs. production distributions)
- Information theory (information gain)
- Machine learning (loss functions, variational inference)
- Statistical inference

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 1 element
- Vectors are treated as probability distributions (will be normalized)
- vectorP is the "true" distribution, vectorQ is the approximation
- All values must be non-negative

Edge cases:
- Panics if vectors have different lengths
- Panics if any value in vectorP is negative
- Panics if any value in vectorQ is negative
- Skips terms where p_i = 0 (0 * log(0/q) = 0)
- Panics if q_i = 0 and p_i > 0 (undefined: log(p/0))
- Returns 0 if distributions are identical
- Range: 0 to ∞ (0 = identical distributions)

Formula: KL(P||Q) = Σ p_i * ln(p_i / q_i)
*/
func StatArchDistanceVectorKLDivergenceF32[T foundation.Numeric](
	analysisP *core.StatArchAnalysis[T],
	analysisQ *core.StatArchAnalysis[T],
) float32 {
	core.StatArchAnalysisValidateVersion(analysisP)
	core.StatArchAnalysisValidateVersion(analysisQ)

	sizeP := analysisP.Count
	sizeQ := analysisQ.Count

	if sizeP != sizeQ {
		panic("vectors must have the same length")
	}
	if sizeP == 0 {
		panic("cannot compute KL divergence with empty vectors")
	}

	// Check for negative values
	memstruct.VectorUnaryReadOnlyExecute(analysisP.Vector, func(item T) {
		val := float32(item)
		if val < 0 {
			panic("KL divergence requires non-negative values in vectorP")
		}
	}, core.StatarchDefaultStride)
	memstruct.VectorUnaryReadOnlyExecute(analysisQ.Vector, func(item T) {
		val := float32(item)
		if val < 0 {
			panic("KL divergence requires non-negative values in vectorQ")
		}
	}, core.StatarchDefaultStride)

	// Use cached sums if available, otherwise compute and cache
	sumP := descriptive.StatArchDescriptiveVectorSumF32(analysisP)
	sumQ := descriptive.StatArchDescriptiveVectorSumF32(analysisQ)

	if sumP == 0 {
		panic("cannot compute KL divergence: vectorP sum is 0")
	}
	if sumQ == 0 {
		panic("cannot compute KL divergence: vectorQ sum is 0")
	}

	// Compute KL divergence
	var klDivergence float32 = 0
	memstruct.VectorBinaryReadOnlyExecute(analysisP.Vector, analysisQ.Vector, func(itemP T, itemQ T) {
		p := float32(itemP) / sumP
		q := float32(itemQ) / sumQ

		// Skip if p = 0 (0 * log(0/q) = 0)
		if p == 0 {
			return
		}

		// Check for undefined case: q = 0 and p > 0
		if q == 0 {
			panic("KL divergence undefined: q_i = 0 and p_i > 0")
		}

		klDivergence += p * foundation.Log32(p/q)
	}, core.StatarchDefaultStride)

	return klDivergence
}

/*
StatArchDistanceVectorKLDivergenceF64 computes the Kullback-Leibler divergence in float64 precision.

KL divergence measures how much one probability distribution diverges from another.
It is asymmetric: KL(P||Q) ≠ KL(Q||P), where P is the "true" distribution and Q is the approximation.

Use cases:
- Detecting model drift (comparing training vs. production distributions)
- Information theory (information gain)
- Machine learning (loss functions, variational inference)
- Statistical inference

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both vectors must have the same length
- Both vectors must contain at least 1 element
- Vectors are treated as probability distributions (will be normalized)
- vectorP is the "true" distribution, vectorQ is the approximation
- All values must be non-negative

Edge cases:
- Panics if vectors have different lengths
- Panics if any value in vectorP is negative
- Panics if any value in vectorQ is negative
- Skips terms where p_i = 0 (0 * log(0/q) = 0)
- Panics if q_i = 0 and p_i > 0 (undefined: log(p/0))
- Returns 0 if distributions are identical
- Range: 0 to ∞ (0 = identical distributions)

Formula: KL(P||Q) = Σ p_i * ln(p_i / q_i)
*/
func StatArchDistanceVectorKLDivergenceF64[T foundation.Numeric](
	analysisP *core.StatArchAnalysis[T],
	analysisQ *core.StatArchAnalysis[T],
) float64 {
	core.StatArchAnalysisValidateVersion(analysisP)
	core.StatArchAnalysisValidateVersion(analysisQ)

	sizeP := analysisP.Count
	sizeQ := analysisQ.Count

	if sizeP != sizeQ {
		panic("vectors must have the same length")
	}
	if sizeP == 0 {
		panic("cannot compute KL divergence with empty vectors")
	}

	// Check for negative values
	memstruct.VectorUnaryReadOnlyExecute(analysisP.Vector, func(item T) {
		val := float64(item)
		if val < 0 {
			panic("KL divergence requires non-negative values in vectorP")
		}
	}, core.StatarchDefaultStride)
	memstruct.VectorUnaryReadOnlyExecute(analysisQ.Vector, func(item T) {
		val := float64(item)
		if val < 0 {
			panic("KL divergence requires non-negative values in vectorQ")
		}
	}, core.StatarchDefaultStride)

	// Use cached sums if available, otherwise compute and cache
	sumP := descriptive.StatArchDescriptiveVectorSumF64(analysisP)
	sumQ := descriptive.StatArchDescriptiveVectorSumF64(analysisQ)

	if sumP == 0 {
		panic("cannot compute KL divergence: vectorP sum is 0")
	}
	if sumQ == 0 {
		panic("cannot compute KL divergence: vectorQ sum is 0")
	}

	// Compute KL divergence
	var klDivergence float64 = 0
	memstruct.VectorBinaryReadOnlyExecute(analysisP.Vector, analysisQ.Vector, func(itemP T, itemQ T) {
		p := float64(itemP) / sumP
		q := float64(itemQ) / sumQ

		// Skip if p = 0 (0 * log(0/q) = 0)
		if p == 0 {
			return
		}

		// Check for undefined case: q = 0 and p > 0
		if q == 0 {
			panic("KL divergence undefined: q_i = 0 and p_i > 0")
		}

		klDivergence += p * foundation.Log64(p/q)
	}, core.StatarchDefaultStride)

	return klDivergence
}



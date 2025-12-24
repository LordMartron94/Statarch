package hypothesis

import (
	"foundation"
	"math"
	"memarch"
	"memcore"
	"memstruct"
	"statarch/core"
	"statarch/descriptive"
)

/*
StatArchHypothesisVectorJarqueBeraF32 computes the Jarque-Bera test statistic in float32 precision.

The Jarque-Bera test is a goodness-of-fit test that determines if sample data has
skewness and kurtosis matching a normal distribution. The test statistic follows
a chi-squared distribution with 2 degrees of freedom.

Use cases:
- Testing normality assumption (required for many statistical tests)
- Quality control (checking if process data is normally distributed)
- Model validation (checking residual normality)
- Statistical inference (pre-test for parametric tests)

Time complexity: O(1) - uses cached skewness and kurtosis from analysis
Space complexity: O(1) - only local variables used

Prerequisites:
- analysis must be a valid StatArchAnalysis structure from statarch/core
- Vector must contain at least 3 elements (population) or 4 elements (sample) for skewness/kurtosis
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population

Edge cases:
- Panics if vector has insufficient elements (inherited from skewness/kurtosis requirements)
- Test statistic is chi-squared distributed with 2 degrees of freedom
- Large values indicate deviation from normality
- Critical values: 5.99 (α=0.05), 9.21 (α=0.01)

Formula: JB = (n/6) * (skewness² + (kurtosis²/4))
*/
func StatArchHypothesisVectorJarqueBeraF32[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float32 {
	core.StatArchAnalysisValidateVersion(analysis)

	skewness := descriptive.StatArchDescriptiveVectorSkewnessF32(analysis, sample)
	kurtosis := descriptive.StatArchDescriptiveVectorKurtosisF32(analysis, sample)

	n := float32(core.StatArchAnalysisCount(analysis))

	// Jarque-Bera test statistic
	skewnessSq := skewness * skewness
	kurtosisSq := kurtosis * kurtosis
	jb := (n / 6.0) * (skewnessSq + (kurtosisSq / 4.0))

	return jb
}

/*
StatArchHypothesisVectorJarqueBeraF64 computes the Jarque-Bera test statistic in float64 precision.

The Jarque-Bera test is a goodness-of-fit test that determines if sample data has
skewness and kurtosis matching a normal distribution. The test statistic follows
a chi-squared distribution with 2 degrees of freedom.

Use cases:
- Testing normality assumption (required for many statistical tests)
- Quality control (checking if process data is normally distributed)
- Model validation (checking residual normality)
- Statistical inference (pre-test for parametric tests)

Time complexity: O(1) - uses cached skewness and kurtosis from analysis
Space complexity: O(1) - only local variables used

Prerequisites:
- analysis must be a valid StatArchAnalysis structure from statarch/core
- Vector must contain at least 3 elements (population) or 4 elements (sample) for skewness/kurtosis
- If sample=true, vector represents a sample from a larger population
- If sample=false, vector represents the entire population

Edge cases:
- Panics if vector has insufficient elements (inherited from skewness/kurtosis requirements)
- Test statistic is chi-squared distributed with 2 degrees of freedom
- Large values indicate deviation from normality
- Critical values: 5.99 (α=0.05), 9.21 (α=0.01)

Formula: JB = (n/6) * (skewness² + (kurtosis²/4))
*/
func StatArchHypothesisVectorJarqueBeraF64[T foundation.Numeric](
	analysis *core.StatArchAnalysis[T],
	sample bool,
) float64 {
	core.StatArchAnalysisValidateVersion(analysis)

	skewness := descriptive.StatArchDescriptiveVectorSkewnessF64(analysis, sample)
	kurtosis := descriptive.StatArchDescriptiveVectorKurtosisF64(analysis, sample)

	n := float64(core.StatArchAnalysisCount(analysis))

	// Jarque-Bera test statistic
	skewnessSq := skewness * skewness
	kurtosisSq := kurtosis * kurtosis
	jb := (n / 6.0) * (skewnessSq + (kurtosisSq / 4.0))

	return jb
}

// MannWhitneyResult holds the results of a Mann-Whitney U test.
type MannWhitneyResult struct {
	UStatistic float64 // The U statistic (minimum of U_A and U_B)
	ZScore     float64 // Z-score for normal approximation (only valid for large samples)
	PValue     float64 // Two-tailed p-value (only valid for large samples)
}

/*
StatArchHypothesisVectorMannWhitneyUF32 computes the Mann-Whitney U test statistic in float32 precision.

The Mann-Whitney U test (also known as Wilcoxon rank-sum test) is a non-parametric test
that determines whether two independent samples come from the same distribution. It is the
standard alternative to the t-test when data is not normally distributed.

Use cases:
- A/B testing (website latency, request sizes, user engagement metrics)
- Comparing non-normal distributions
- Quality control (comparing process outputs)
- Medical research (comparing treatment groups with non-normal outcomes)
- When normality assumption fails (use after Jarque-Bera test rejects normality)

Time complexity: O(n log n) - dominated by ranking operation
Space complexity: O(n) - requires temporary storage for combined ranking

Prerequisites:
- Both analysis structures must be valid
- Both vectors must contain at least 1 element
- Vectors do not need to be sorted
- allocFn must be provided for creating temporary vectors

Edge cases:
- Panics if either vector has < 1 element
- For small samples (n_A + n_B ≤ 20), exact distribution should be used (not implemented)
- For large samples, uses normal approximation
- If all values are identical, U = n_A*n_B/2, p = 1.0

Algorithm:
1. Combine both groups into one vector
2. Rank all values together (handling ties with average ranks)
3. Compute rank sums: R_A = sum of ranks in group A, R_B = sum of ranks in group B
4. Compute U statistics: U_A = n_A*n_B + n_A*(n_A+1)/2 - R_A, U_B = n_A*n_B - U_A
5. Use smaller U (U_min = min(U_A, U_B))
6. For large samples (n_A + n_B > 20): Normal approximation
  - μ_U = n_A*n_B/2
  - σ_U = sqrt(n_A*n_B*(n_A+n_B+1)/12) with tie correction
  - z = (U_min - μ_U) / σ_U

7. Compute two-tailed p-value from z-score

Ties handling: Adjusts variance for ties using correction factor:
σ_U² = n_A*n_B/(n*(n-1)) * (n³-n - Σt(t²-1))/12 where t is tie group size

The function returns U statistic, z-score, and p-value. For small samples, the p-value
may not be accurate (exact distribution computation would be needed).
*/
func StatArchHypothesisVectorMannWhitneyUF32[T foundation.Numeric](
	analysisA *core.StatArchAnalysis[T],
	analysisB *core.StatArchAnalysis[T],
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) MannWhitneyResult {
	core.StatArchAnalysisValidateVersion(analysisA)
	core.StatArchAnalysisValidateVersion(analysisB)

	nA := analysisA.Count
	nB := analysisB.Count

	if nA < 1 || nB < 1 {
		panic("Mann-Whitney U test requires at least 1 element in each group")
	}

	n := nA + nB

	// Step 1: Combine both groups into one vector
	combinedVector, _ := memarch.MemArchVectorCreate[T](allocFn, n)
	var idx uint64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysisA.Vector, func(item T) {
		memstruct.VectorSetAtUnsafe(combinedVector, idx, item)
		idx++
	}, 1)
	memstruct.VectorUnaryReadOnlyExecute(analysisB.Vector, func(item T) {
		memstruct.VectorSetAtUnsafe(combinedVector, idx, item)
		idx++
	}, 1)

	// Step 2: Rank all values together
	rankVector, _ := memarch.MemArchVectorCreate[float32](allocFn, n)
	rankVectorF32[T](rankVector, combinedVector, n, allocFn)

	// Step 3: Compute rank sums
	var rankSumA float64 = 0
	for i := uint64(0); i < nA; i++ {
		rank := float64(memstruct.VectorItemGetAtUnsafe[float32](rankVector, i))
		rankSumA += rank
	}

	var rankSumB float64 = 0
	for i := uint64(0); i < nB; i++ {
		rank := float64(memstruct.VectorItemGetAtUnsafe[float32](rankVector, nA+i))
		rankSumB += rank
	}

	// Step 4: Compute U statistics
	uA := float64(nA)*float64(nB) + float64(nA)*float64(nA+1)/2.0 - rankSumA
	uB := float64(nA)*float64(nB) + float64(nB)*float64(nB+1)/2.0 - rankSumB
	uMin := uA
	if uB < uA {
		uMin = uB
	}

	// Step 5: Check if all values are identical (all ranks equal)
	firstRank := memstruct.VectorItemGetAtUnsafe[float32](rankVector, 0)
	allIdentical := true
	for i := uint64(1); i < n; i++ {
		rank := memstruct.VectorItemGetAtUnsafe[float32](rankVector, i)
		if rank != firstRank {
			allIdentical = false
			break
		}
	}

	if allIdentical {
		// All values identical: U = n_A*n_B/2, p = 1.0
		return MannWhitneyResult{
			UStatistic: float64(nA) * float64(nB) / 2.0,
			ZScore:     0.0,
			PValue:     1.0,
		}
	}

	// Step 6: For large samples, use normal approximation
	if n <= 20 {
		// Small sample: exact distribution would be needed
		// For now, return U statistic with NaN for z-score and p-value
		return MannWhitneyResult{
			UStatistic: uMin,
			ZScore:     math.NaN(),
			PValue:     math.NaN(),
		}
	}

	// Compute tie correction
	// Count ties: group by rank value
	tieCorrection := 0.0
	rankCounts := make(map[float32]uint64)
	for i := uint64(0); i < n; i++ {
		rank := memstruct.VectorItemGetAtUnsafe[float32](rankVector, i)
		rankCounts[rank]++
	}

	// Compute Σt(t²-1) where t is tie group size
	for _, count := range rankCounts {
		if count > 1 {
			tieCorrection += float64(count) * (float64(count)*float64(count) - 1.0)
		}
	}

	// Mean and variance with tie correction
	muU := float64(nA) * float64(nB) / 2.0
	var sigmaUSq float64
	if tieCorrection > 0 {
		// With ties: σ_U² = n_A*n_B/(n*(n-1)) * (n³-n - Σt(t²-1))/12
		n3 := float64(n) * float64(n) * float64(n)
		sigmaUSq = (float64(nA) * float64(nB) / (float64(n) * float64(n-1))) * ((n3 - float64(n) - tieCorrection) / 12.0)
	} else {
		// No ties: σ_U² = n_A*n_B*(n_A+n_B+1)/12
		sigmaUSq = float64(nA) * float64(nB) * float64(n+1) / 12.0
	}
	sigmaU := math.Sqrt(sigmaUSq)

	// Z-score
	zScore := (uMin - muU) / sigmaU

	// Step 7: Compute two-tailed p-value from z-score
	// P-value = 2 * (1 - Φ(|z|)) where Φ is standard normal CDF
	pValue := 2.0 * (1.0 - normalCDFApprox(math.Abs(zScore)))

	return MannWhitneyResult{
		UStatistic: uMin,
		ZScore:     zScore,
		PValue:     pValue,
	}
}

/*
StatArchHypothesisVectorMannWhitneyUF64 computes the Mann-Whitney U test statistic in float64 precision.

See StatArchHypothesisVectorMannWhitneyUF32 for detailed documentation.
This is the float64 variant for higher precision requirements.
*/
func StatArchHypothesisVectorMannWhitneyUF64[T foundation.Numeric](
	analysisA *core.StatArchAnalysis[T],
	analysisB *core.StatArchAnalysis[T],
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) MannWhitneyResult {
	core.StatArchAnalysisValidateVersion(analysisA)
	core.StatArchAnalysisValidateVersion(analysisB)

	nA := analysisA.Count
	nB := analysisB.Count

	if nA < 1 || nB < 1 {
		panic("Mann-Whitney U test requires at least 1 element in each group")
	}

	n := nA + nB

	// Step 1: Combine both groups into one vector
	combinedVector, _ := memarch.MemArchVectorCreate[T](allocFn, n)
	var idx uint64 = 0
	memstruct.VectorUnaryReadOnlyExecute(analysisA.Vector, func(item T) {
		memstruct.VectorSetAtUnsafe(combinedVector, idx, item)
		idx++
	}, 1)
	memstruct.VectorUnaryReadOnlyExecute(analysisB.Vector, func(item T) {
		memstruct.VectorSetAtUnsafe(combinedVector, idx, item)
		idx++
	}, 1)

	// Step 2: Rank all values together
	rankVector, _ := memarch.MemArchVectorCreate[float64](allocFn, n)
	rankVectorF64[T](rankVector, combinedVector, n, allocFn)

	// Step 3: Compute rank sums
	var rankSumA float64 = 0
	for i := uint64(0); i < nA; i++ {
		rank := memstruct.VectorItemGetAtUnsafe[float64](rankVector, i)
		rankSumA += rank
	}

	var rankSumB float64 = 0
	for i := uint64(0); i < nB; i++ {
		rank := memstruct.VectorItemGetAtUnsafe[float64](rankVector, nA+i)
		rankSumB += rank
	}

	// Step 4: Compute U statistics
	uA := float64(nA)*float64(nB) + float64(nA)*float64(nA+1)/2.0 - rankSumA
	uB := float64(nA)*float64(nB) + float64(nB)*float64(nB+1)/2.0 - rankSumB
	uMin := uA
	if uB < uA {
		uMin = uB
	}

	// Step 5: Check if all values are identical (all ranks equal)
	firstRank := memstruct.VectorItemGetAtUnsafe[float64](rankVector, 0)
	allIdentical := true
	for i := uint64(1); i < n; i++ {
		rank := memstruct.VectorItemGetAtUnsafe[float64](rankVector, i)
		if rank != firstRank {
			allIdentical = false
			break
		}
	}

	if allIdentical {
		// All values identical: U = n_A*n_B/2, p = 1.0
		return MannWhitneyResult{
			UStatistic: float64(nA) * float64(nB) / 2.0,
			ZScore:     0.0,
			PValue:     1.0,
		}
	}

	// Step 6: For large samples, use normal approximation
	if n <= 20 {
		// Small sample: exact distribution would be needed
		// For now, return U statistic with NaN for z-score and p-value
		return MannWhitneyResult{
			UStatistic: uMin,
			ZScore:     math.NaN(),
			PValue:     math.NaN(),
		}
	}

	// Compute tie correction
	// Count ties: group by rank value
	tieCorrection := 0.0
	rankCounts := make(map[float64]uint64)
	for i := uint64(0); i < n; i++ {
		rank := memstruct.VectorItemGetAtUnsafe[float64](rankVector, i)
		rankCounts[rank]++
	}

	// Compute Σt(t²-1) where t is tie group size
	for _, count := range rankCounts {
		if count > 1 {
			tieCorrection += float64(count) * (float64(count)*float64(count) - 1.0)
		}
	}

	// Mean and variance with tie correction
	muU := float64(nA) * float64(nB) / 2.0
	var sigmaUSq float64
	if tieCorrection > 0 {
		// With ties: σ_U² = n_A*n_B/(n*(n-1)) * (n³-n - Σt(t²-1))/12
		n3 := float64(n) * float64(n) * float64(n)
		sigmaUSq = (float64(nA) * float64(nB) / (float64(n) * float64(n-1))) * ((n3 - float64(n) - tieCorrection) / 12.0)
	} else {
		// No ties: σ_U² = n_A*n_B*(n_A+n_B+1)/12
		sigmaUSq = float64(nA) * float64(nB) * float64(n+1) / 12.0
	}
	sigmaU := math.Sqrt(sigmaUSq)

	// Z-score
	zScore := (uMin - muU) / sigmaU

	// Step 7: Compute two-tailed p-value from z-score
	// P-value = 2 * (1 - Φ(|z|)) where Φ is standard normal CDF
	pValue := 2.0 * (1.0 - normalCDFApprox(math.Abs(zScore)))

	return MannWhitneyResult{
		UStatistic: uMin,
		ZScore:     zScore,
		PValue:     pValue,
	}
}

// rankEntryF32 binds a value with its original index for ranking purposes in float32 precision.
type rankEntryF32[T foundation.Numeric] struct {
	Value float32
	Index uint64
}

// rankVectorF32 computes ranks for a vector in float32, handling ties with average ranks.
// This is a copy of the ranking logic from correlation package, adapted for hypothesis testing.
func rankVectorF32[T foundation.Numeric](
	rankVector memcore.MarkRaw,
	sourceVector memcore.MarkRaw,
	size uint64,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) {
	// Create Array of rankEntryF32 structs
	entries, _ := memarch.MemArchArrayCreate[rankEntryF32[T]](allocFn, size)

	// Fill: Pass through the source vector once, filling entries with value and index
	var idx uint64 = 0
	memstruct.VectorUnaryReadOnlyExecute(sourceVector, func(item T) {
		entry := rankEntryF32[T]{
			Value: float32(item),
			Index: idx,
		}
		memstruct.ArraySetAtUnsafe(entries, idx, entry)
		idx++
	}, 1)

	// Sort: Call ArraySort on the entry array. The comparator only looks at Entry.Value.
	memstruct.ArraySort(entries, func(a, b rankEntryF32[T]) int {
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
		currentEntry := memstruct.ArrayItemGetAtUnsafe[rankEntryF32[T]](entries, i)
		currentVal := currentEntry.Value

		for j := i + 1; j < size; j++ {
			nextEntry := memstruct.ArrayItemGetAtUnsafe[rankEntryF32[T]](entries, j)
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
			entry := memstruct.ArrayItemGetAtUnsafe[rankEntryF32[T]](entries, i+j)
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
// This is a copy of the ranking logic from correlation package, adapted for hypothesis testing.
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

// normalCDFApprox approximates the standard normal CDF using the error function.
// Φ(x) = 0.5 * (1 + erf(x/√2))
// This provides good accuracy for hypothesis testing purposes.
func normalCDFApprox(z float64) float64 {
	if z < 0 {
		return 1.0 - normalCDFApprox(-z)
	}
	if z > 6 {
		return 1.0
	}
	// Use error function: erf(x) = 2/√π * ∫[0 to x] e^(-t²) dt
	// Φ(x) = 0.5 * (1 + erf(x/√2))
	return 0.5 * (1.0 + math.Erf(z/math.Sqrt2))
}

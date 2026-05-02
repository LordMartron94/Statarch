package hypothesis

import (
	"math"
	"memarch"
	"memcore"
	"memforge"
	"memstruct"
	"statarch/core"
	"statarch/descriptive"
	"testing"
)

func TestStatArchHypothesisVectorConfidenceIntervalF64(t *testing.T) {
	allocator := memforge.DynamicLinearAllocatorCreateFunction(1024, func(currentCap, needed uint64) uint64 {
		newSize := currentCap * 2
		if newSize < needed {
			newSize = needed
		}
		return newSize
	})
	defer memforge.DynamicLinearAllocatorDestroy(allocator)

	allocFn := func(sizeBytes, alignment uint64) memcore.MarkRaw {
		return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
	}

	// Test with known data: [1, 2, 3, 4, 5]
	// Mean = 3.0, StdDev ≈ 1.58, SEM ≈ 0.707
	vector, _ := memarch.MemArchVectorCreate[float64](allocFn, 5)
	memstruct.VectorSetFromSlice(vector, []float64{1.0, 2.0, 3.0, 4.0, 5.0})

	analysis := core.StatArchAnalysisCreate[float64](vector, allocFn)

	// Test 95% confidence interval
	lower, upper := StatArchHypothesisVectorConfidenceIntervalF64(analysis, 0.95, true)

	mean := descriptive.StatArchDescriptiveVectorMeanF64(analysis)
	if mean != 3.0 {
		t.Errorf("Expected mean 3.0, got %f", mean)
	}

	// Confidence interval should contain the mean
	if lower > mean || upper < mean {
		t.Errorf("Confidence interval [%f, %f] should contain mean %f", lower, upper, mean)
	}

	// Lower bound should be less than upper bound
	if lower >= upper {
		t.Errorf("Lower bound %f should be less than upper bound %f", lower, upper)
	}

	// For 95% confidence with n=5, interval should be reasonably wide
	// Expected: mean ± t_critical * SEM
	// With t_critical ≈ 2.776 for df=4 at 95%, SEM ≈ 0.707
	// Expected margin ≈ 1.96, so interval should be roughly [1.04, 4.96]
	// We'll check that it's in a reasonable range
	if lower < 0.5 || upper > 5.5 {
		t.Errorf("Confidence interval [%f, %f] seems unreasonable for data [1,2,3,4,5]", lower, upper)
	}
}

func TestStatArchHypothesisVectorConfidenceIntervalF64_EmptyVector(t *testing.T) {
	allocator := memforge.DynamicLinearAllocatorCreateFunction(1024, func(currentCap, needed uint64) uint64 {
		return needed
	})
	defer memforge.DynamicLinearAllocatorDestroy(allocator)

	allocFn := func(sizeBytes, alignment uint64) memcore.MarkRaw {
		return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
	}

	vector, _ := memarch.MemArchVectorCreate[float64](allocFn, 0)
	analysis := core.StatArchAnalysisCreate[float64](vector, allocFn)

	lower, upper := StatArchHypothesisVectorConfidenceIntervalF64(analysis, 0.95, true)

	// Should return [mean, mean] for empty vector
	// The function checks Count < 2, so it should return mean, mean
	mean := descriptive.StatArchDescriptiveVectorMeanF64(analysis)
	// For empty vector, mean might be NaN or 0, but lower and upper should match mean
	if (math.IsNaN(lower) && !math.IsNaN(mean)) || (math.IsNaN(upper) && !math.IsNaN(mean)) {
		t.Errorf("Lower and upper should match mean for empty vector, got [%f, %f], mean=%f", lower, upper, mean)
	}
	if !math.IsNaN(mean) && (lower != mean || upper != mean) {
		t.Errorf("Expected [%f, %f] for empty vector, got [%f, %f]", mean, mean, lower, upper)
	}
}

func TestStatArchHypothesisVectorConfidenceIntervalF64_SingleElement(t *testing.T) {
	allocator := memforge.DynamicLinearAllocatorCreateFunction(1024, func(currentCap, needed uint64) uint64 {
		return needed
	})
	defer memforge.DynamicLinearAllocatorDestroy(allocator)

	allocFn := func(sizeBytes, alignment uint64) memcore.MarkRaw {
		return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
	}

	vector, _ := memarch.MemArchVectorCreate[float64](allocFn, 1)
	memstruct.VectorSetFromSlice(vector, []float64{42.0})
	analysis := core.StatArchAnalysisCreate[float64](vector, allocFn)

	lower, upper := StatArchHypothesisVectorConfidenceIntervalF64(analysis, 0.95, true)

	// Should return [mean, mean] for single element
	if lower != 42.0 || upper != 42.0 {
		t.Errorf("Expected [42.0, 42.0] for single element, got [%f, %f]", lower, upper)
	}
}

func TestComputeTCritical(t *testing.T) {
	// Test that t-critical values are reasonable
	// For large samples (n >= 30), should use z-score
	critical := computeTCritical(30.0, 0.95)
	if critical < 1.9 || critical > 2.1 {
		t.Errorf("Expected t-critical around 1.96 for n=30, 95%% confidence, got %f", critical)
	}

	// For small samples, should use t-distribution
	critical = computeTCritical(10.0, 0.95)
	if critical < 2.0 || critical > 2.5 {
		t.Errorf("Expected t-critical around 2.26 for n=10, 95%% confidence, got %f", critical)
	}

	// For 99% confidence, should be larger
	critical99 := computeTCritical(10.0, 0.99)
	if critical99 <= critical {
		t.Errorf("Expected higher t-critical for 99%% confidence, got %f vs %f", critical99, critical)
	}
}

func TestComputeZCritical(t *testing.T) {
	// Test z-critical values
	z90 := computeZCritical(0.90)
	if math.Abs(z90-1.645) > 0.01 {
		t.Errorf("Expected z-critical 1.645 for 90%% confidence, got %f", z90)
	}

	z95 := computeZCritical(0.95)
	if math.Abs(z95-1.96) > 0.01 {
		t.Errorf("Expected z-critical 1.96 for 95%% confidence, got %f", z95)
	}

	z99 := computeZCritical(0.99)
	if math.Abs(z99-2.576) > 0.01 {
		t.Errorf("Expected z-critical 2.576 for 99%% confidence, got %f", z99)
	}

	// 99% should be larger than 95%
	if z99 <= z95 {
		t.Errorf("Expected z99 > z95, got %f vs %f", z99, z95)
	}
}

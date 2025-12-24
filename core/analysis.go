package core

import (
	"foundation"
	"memarch"
	"memcore"
	"memstruct"
)

// StatKind represents the type of statistic being cached.
type StatKind uint64

// Constants for each statistic kind.
const (
	StatKindMeanF32 StatKind = iota + 1
	StatKindMeanF64
	StatKindHarmonicMeanF32
	StatKindHarmonicMeanF64
	StatKindGeometricMeanF32
	StatKindGeometricMeanF64
	StatKindMedianF32
	StatKindMedianF64
	StatKindMode
	StatKindModeOccurrence
	StatKindVarianceF32Sample
	StatKindVarianceF32Population
	StatKindVarianceF64Sample
	StatKindVarianceF64Population
	StatKindStddevF32Sample
	StatKindStddevF32Population
	StatKindStddevF64Sample
	StatKindStddevF64Population
	StatKindMin
	StatKindMax
	StatKindIQRF32
	StatKindIQRF64
	StatKindNormalizedIQRF32
	StatKindNormalizedIQRF64
	StatKindSkewnessF32Sample
	StatKindSkewnessF32Population
	StatKindSkewnessF64Sample
	StatKindSkewnessF64Population
	StatKindKurtosisF32Sample
	StatKindKurtosisF32Population
	StatKindKurtosisF64Sample
	StatKindKurtosisF64Population
	StatKindMeanAbsoluteDeviationF32
	StatKindMeanAbsoluteDeviationF64
	StatKindMedianAbsoluteDeviationF32
	StatKindMedianAbsoluteDeviationF64
	StatKindCoefficientVariationF32Sample
	StatKindCoefficientVariationF32Population
	StatKindCoefficientVariationF64Sample
	StatKindCoefficientVariationF64Population
	StatKindStandardErrorMeanF32Sample
	StatKindStandardErrorMeanF32Population
	StatKindStandardErrorMeanF64Sample
	StatKindStandardErrorMeanF64Population
	StatKindWinsorizedMeanF32
	StatKindWinsorizedMeanF64
	StatKindMidRangeF32
	StatKindMidRangeF64
	StatKindBimodalityCoefficientF32
	StatKindBimodalityCoefficientF64
	StatKindRollingMeanF32
	StatKindRollingMeanF64
	StatKindRollingVarianceF32Sample
	StatKindRollingVarianceF32Population
	StatKindRollingVarianceF64Sample
	StatKindRollingVarianceF64Population
	StatKindRollingStddevF32Sample
	StatKindRollingStddevF32Population
	StatKindRollingStddevF64Sample
	StatKindRollingStddevF64Population
	StatKindRollingMin
	StatKindRollingMax
	StatKindRollingSum
	StatKindRollingSkewnessF32Sample
	StatKindRollingSkewnessF32Population
	StatKindRollingSkewnessF64Sample
	StatKindRollingSkewnessF64Population
	StatKindRollingKurtosisF32Sample
	StatKindRollingKurtosisF32Population
	StatKindRollingKurtosisF64Sample
	StatKindRollingKurtosisF64Population
)

/*
StatArchAnalysis holds a vector and its computed descriptive statistics.

Functions operate on this structure, computing and storing values as needed to avoid
redundant calculations. For example, if variance is computed, it will use the cached mean
value rather than recomputing it.

Time complexity: O(1) for accessing cached values, O(n) for computing new values
Space complexity: O(k) where k is the number of cached statistics (typically small)

Use cases:
- Computing multiple statistics on the same vector efficiently
- Avoiding redundant mean/variance calculations
- Performance optimization when computing multiple descriptive statistics

The structure caches computed values in a map, allowing for unlimited statistics
as the library grows without being constrained by bit flags.

The structure distinguishes between sample and population statistics (variance, standard deviation)
and caches them separately to ensure accuracy.

Cache Invalidation:
The structure automatically detects when the underlying vector has been modified using version tracking.
Each vector maintains a version number that increments on every modification. The analysis structure
stores the vector's version when created (SourceVersion). Before returning any cached value, the
analysis validates that the current vector version matches SourceVersion. If versions don't match,
the entire cache is invalidated and SourceVersion is updated. This ensures that cached statistics
are never stale, even in long-running processes or streaming data scenarios.
*/
type StatArchAnalysis[T foundation.Numeric] struct {
	Vector memcore.MarkRaw

	// Cached computed values stored as interface{} to support different types (float32, float64, T)
	// Keys are StatKind constants, values are the computed statistics
	Cache map[StatKind]interface{}

	// Sorted copy of the vector (created on demand for functions that require sorted data)
	SortedVector  memcore.MarkRaw
	SortedCreated bool

	// Scratch vector for operations that need temporary storage (e.g., Mode)
	ScratchVector  memcore.MarkRaw
	ScratchCreated bool

	// Scratch vectors for float32/float64 operations (e.g., MAD)
	ScratchVectorF32  memcore.MarkRaw
	ScratchCreatedF32 bool
	ScratchVectorF64  memcore.MarkRaw
	ScratchCreatedF64 bool

	// Function to allocate memory for sorted/scratch vectors
	AllocFn func(sizeBytes, alignment uint64) memcore.MarkRaw

	Count uint64

	// SourceVersion stores the version of the vector when the analysis was created.
	// This is used to detect when the underlying vector has been modified, requiring
	// cache invalidation to prevent returning stale cached statistics.
	SourceVersion uint64
}

/*
StatArchAnalysisCreate creates a new analysis structure for a vector.

Time complexity: O(1) - only initializes structure and gets vector capacity
Space complexity: O(1) - initializes empty map

Parameters:
- vector: The vector to analyze (must be valid and initialized)
- allocFn: Function to allocate memory for sorted/scratch vectors when needed

Returns:
- A new analysis structure ready for use with descriptive statistics functions

The structure is initialized with:
- Vector reference stored
- Count set to vector capacity
- Empty cache map (no values cached yet)
- Sorted and scratch vectors are nil (created on demand)
- SourceVersion set to the current vector version for cache invalidation tracking
*/
func StatArchAnalysisCreate[T foundation.Numeric](
	vector memcore.MarkRaw,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) *StatArchAnalysis[T] {
	return &StatArchAnalysis[T]{
		Vector:        vector,
		Cache:         make(map[StatKind]interface{}),
		AllocFn:       allocFn,
		Count:         memstruct.VectorCapacityGet[T](vector),
		SourceVersion: memstruct.VectorVersionGet[T](vector),
	}
}

/*
StatArchAnalysisCount returns the number of elements in the analysis vector.

This is a helper function to access the count from other packages since the count field is exported.
*/
func StatArchAnalysisCount[T foundation.Numeric](analysis *StatArchAnalysis[T]) uint64 {
	return analysis.Count
}

// GetSortedVector returns a sorted copy of the vector, creating it if necessary.
// Validates version before returning cached sorted vector to ensure it's not stale.
func GetSortedVector[T foundation.Numeric](analysis *StatArchAnalysis[T]) memcore.MarkRaw {
	StatArchAnalysisValidateVersion(analysis)

	if analysis.SortedCreated {
		return analysis.SortedVector
	}

	// Create sorted vector from original
	analysis.SortedVector, _ = memarch.MemArchVectorCreateFrom[T](
		analysis.AllocFn,
		analysis.Vector,
		analysis.Count,
	)
	memstruct.VectorSort(analysis.SortedVector, func(a, b T) int {
		if a < b {
			return -1
		}
		if a > b {
			return 1
		}
		return 0
	})

	analysis.SortedCreated = true
	return analysis.SortedVector
}

// GetScratchVector returns a scratch vector for temporary operations, creating it if necessary.
// Validates version before returning cached scratch vector to ensure it's not stale.
func GetScratchVector[T foundation.Numeric](analysis *StatArchAnalysis[T]) memcore.MarkRaw {
	StatArchAnalysisValidateVersion(analysis)

	if analysis.ScratchCreated {
		return analysis.ScratchVector
	}

	// Create scratch vector
	analysis.ScratchVector, _ = memarch.MemArchVectorCreate[T](
		analysis.AllocFn,
		analysis.Count,
	)

	analysis.ScratchCreated = true
	return analysis.ScratchVector
}

// GetScratchVectorF32 returns a float32 scratch vector for temporary operations, creating it if necessary.
// Validates version before returning cached scratch vector to ensure it's not stale.
func GetScratchVectorF32[T foundation.Numeric](analysis *StatArchAnalysis[T]) memcore.MarkRaw {
	StatArchAnalysisValidateVersion(analysis)

	if analysis.ScratchCreatedF32 {
		return analysis.ScratchVectorF32
	}

	// Create float32 scratch vector
	analysis.ScratchVectorF32, _ = memarch.MemArchVectorCreate[float32](
		analysis.AllocFn,
		analysis.Count,
	)

	analysis.ScratchCreatedF32 = true
	return analysis.ScratchVectorF32
}

// GetScratchVectorF64 returns a float64 scratch vector for temporary operations, creating it if necessary.
// Validates version before returning cached scratch vector to ensure it's not stale.
func GetScratchVectorF64[T foundation.Numeric](analysis *StatArchAnalysis[T]) memcore.MarkRaw {
	StatArchAnalysisValidateVersion(analysis)

	if analysis.ScratchCreatedF64 {
		return analysis.ScratchVectorF64
	}

	// Create float64 scratch vector
	analysis.ScratchVectorF64, _ = memarch.MemArchVectorCreate[float64](
		analysis.AllocFn,
		analysis.Count,
	)

	analysis.ScratchCreatedF64 = true
	return analysis.ScratchVectorF64
}

/*
StatArchAnalysisInvalidateCache clears all cached statistics and resets the analysis structure.

This function is called when the underlying vector has been modified (version mismatch detected).
It clears the cache map, resets all scratch/sorted vector flags, and updates SourceVersion
to the current vector version.

Time complexity: O(k) where k is the number of cached statistics (typically small)
Space complexity: O(1) - clears existing cache, doesn't allocate new memory
*/
func StatArchAnalysisInvalidateCache[T foundation.Numeric](analysis *StatArchAnalysis[T]) {
	// Clear the cache map
	for k := range analysis.Cache {
		delete(analysis.Cache, k)
	}

	// Reset sorted and scratch vector flags
	analysis.SortedCreated = false
	analysis.ScratchCreated = false
	analysis.ScratchCreatedF32 = false
	analysis.ScratchCreatedF64 = false

	// Update SourceVersion to current vector version
	analysis.SourceVersion = memstruct.VectorVersionGet[T](analysis.Vector)
}

/*
StatArchAnalysisValidateVersion checks if the analysis structure's SourceVersion matches
the current vector version.

If versions match, returns true and the cache is valid.
If versions don't match, invalidates the cache, updates SourceVersion, and returns false.

Time complexity: O(1) - single version comparison
Space complexity: O(1) - no allocations

Returns:
- true if versions match (cache is valid)
- false if versions don't match (cache was invalidated)
*/
func StatArchAnalysisValidateVersion[T foundation.Numeric](analysis *StatArchAnalysis[T]) bool {
	currentVersion := memstruct.VectorVersionGet[T](analysis.Vector)

	if analysis.SourceVersion == currentVersion {
		return true
	}

	// Version mismatch - invalidate cache
	StatArchAnalysisInvalidateCache(analysis)
	return false
}



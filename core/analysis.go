package core

import (
	"foundation"
	"memarch"
	"memcore"
	"memstruct"
	"unsafe"
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
	StatKindSumF32
	StatKindSumF64
	StatKindNormSquaredF32
	StatKindNormSquaredF64
	StatKindNormF32
	StatKindNormF64

	// StatKindCount is the total number of statistic kinds.
	// Since StatKind starts at iota + 1 (value 1), this represents the last StatKind value.
	StatKindCount = StatKindNormF64
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

	// Type-specific caches to avoid convT32/convT64 overhead
	// Indexed by (StatKind - 1) since StatKind starts at 1
	// Size is StatKindCount, allocated once at creation
	// Access through type-specific GetCacheValue and SetCacheValue methods
	cacheF32   []float32
	cacheF64   []float64
	cacheOther []interface{} // For non-float types (T, uint): Min, Max, Mode, ModeOccurrence

	// Validity bitmasks to track which cache entries are valid (needed to distinguish 0.0 from "not cached")
	// Each bit represents whether the corresponding cache entry is valid
	// We need 2 uint64s to cover StatKindCount (96) entries
	cacheF32Valid [2]uint64
	cacheF64Valid [2]uint64

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
		cacheF32:      make([]float32, StatKindCount),
		cacheF64:      make([]float64, StatKindCount),
		cacheOther:    make([]interface{}, StatKindCount),
		AllocFn:       allocFn,
		Count:         memstruct.VectorCapacityGet[T](vector),
		SourceVersion: memstruct.VectorVersionGet[T](vector),
		// cacheF32Valid and cacheF64Valid are zero-initialized (all bits 0 = not cached)
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
	// Clear validity bitmasks
	analysis.cacheF32Valid[0] = 0
	analysis.cacheF32Valid[1] = 0
	analysis.cacheF64Valid[0] = 0
	analysis.cacheF64Valid[1] = 0

	if len(analysis.cacheOther) > 0 {
		memclrHasPointers(unsafe.Pointer(&analysis.cacheOther[0]), uintptr(len(analysis.cacheOther))*unsafe.Sizeof(analysis.cacheOther[0]))
	}

	analysis.SortedCreated = false
	analysis.ScratchCreated = false
	analysis.ScratchCreatedF32 = false
	analysis.ScratchCreatedF64 = false

	analysis.SourceVersion = memstruct.VectorVersionGet[T](analysis.Vector)
}

/*
StatArchAnalysisReset resets an analysis structure to be reused with a new vector.

This function clears the cache map (reusing the existing map to avoid allocations),
resets all scratch/sorted vector flags, and updates the vector reference, count, and
source version. The AllocFn is preserved. This allows reusing a single analysis object
across multiple vectors in hot loops, eliminating expensive allocations.

Use cases:
- Reusing analysis objects in hot loops (e.g., similarity search over many vectors)
- Avoiding allocations when processing multiple vectors sequentially
- Performance optimization in batch operations

Time complexity: O(k) where k is the number of cached statistics (typically small)
Space complexity: O(1) - reuses existing cache map, no new allocations

Parameters:
- analysis: The analysis structure to reset (must be previously created)
- vector: The new vector to analyze (must be valid and initialized)

Prerequisites:
- analysis must be a valid StatArchAnalysis created with StatArchAnalysisCreate
- vector must be a valid MarkRaw pointing to a bound vector header
- AllocFn in analysis must remain valid for the lifetime of the analysis object

Edge cases:
- If the cache map is nil (should not happen with properly created analysis), it will be initialized
- Resetting does not free scratch/sorted vectors; they remain allocated until garbage collected
- SourceVersion is updated to match the new vector's version

Additional notes:
  - This function is designed for hot loop optimization where the same analysis object
    is reused across many iterations. For one-off analysis, StatArchAnalysisCreate is preferred.
  - The cache map is reused, not reallocated, which is the key performance benefit.
*/
func StatArchAnalysisReset[T foundation.Numeric](
	analysis *StatArchAnalysis[T],
	vector memcore.MarkRaw,
) {
	StatArchAnalysisInvalidateCache(analysis)

	// Update vector reference and metadata
	analysis.Vector = vector
	analysis.Count = memstruct.VectorCapacityGet[T](vector)
	analysis.SourceVersion = memstruct.VectorVersionGet[T](vector)
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

/*
StatArchAnalysisGetCacheValueF32 retrieves a cached float32 value by StatKind.

Returns the value and true if cached, 0 and false otherwise.
This method handles the index conversion (StatKind - 1) internally.
Direct access without type assertion overhead.
Uses validity bitmask to track which entries are cached.

Time complexity: O(1)
Space complexity: O(1)
*/
func StatArchAnalysisGetCacheValueF32[T foundation.Numeric](analysis *StatArchAnalysis[T], kind StatKind) (float32, bool) {
	if kind < 1 || kind > StatKindCount {
		return 0, false
	}
	idx := kind - 1
	// Check validity bitmask
	bitIdx := idx / 64
	bitPos := idx % 64
	if bitIdx < 2 && (analysis.cacheF32Valid[bitIdx]&(1<<bitPos)) == 0 {
		return 0, false
	}
	return analysis.cacheF32[idx], true
}

/*
StatArchAnalysisGetCacheValueF64 retrieves a cached float64 value by StatKind.

Returns the value and true if cached, 0 and false otherwise.
This method handles the index conversion (StatKind - 1) internally.
Direct access without type assertion overhead.
Uses validity bitmask to track which entries are cached.

Time complexity: O(1)
Space complexity: O(1)
*/
func StatArchAnalysisGetCacheValueF64[T foundation.Numeric](analysis *StatArchAnalysis[T], kind StatKind) (float64, bool) {
	if kind < 1 || kind > StatKindCount {
		return 0, false
	}
	idx := kind - 1
	// Check validity bitmask
	bitIdx := idx / 64
	bitPos := idx % 64
	if bitIdx < 2 && (analysis.cacheF64Valid[bitIdx]&(1<<bitPos)) == 0 {
		return 0, false
	}
	return analysis.cacheF64[idx], true
}

/*
StatArchAnalysisSetCacheValueF32 sets a cached float32 value by StatKind.

This method handles the index conversion (StatKind - 1) internally.
Direct assignment without type assertion overhead.

Time complexity: O(1)
Space complexity: O(1)
*/
func StatArchAnalysisSetCacheValueF32[T foundation.Numeric](analysis *StatArchAnalysis[T], kind StatKind, value float32) {
	if kind < 1 || kind > StatKindCount {
		return
	}
	idx := kind - 1
	analysis.cacheF32[idx] = value
	// Set validity bit
	bitIdx := idx / 64
	bitPos := idx % 64
	if bitIdx < 2 {
		analysis.cacheF32Valid[bitIdx] |= 1 << bitPos
	}
}

/*
StatArchAnalysisSetCacheValueF64 sets a cached float64 value by StatKind.

This method handles the index conversion (StatKind - 1) internally.
Direct assignment without type assertion overhead.

Time complexity: O(1)
Space complexity: O(1)
*/
func StatArchAnalysisSetCacheValueF64[T foundation.Numeric](analysis *StatArchAnalysis[T], kind StatKind, value float64) {
	if kind < 1 || kind > StatKindCount {
		return
	}
	idx := kind - 1
	analysis.cacheF64[idx] = value
	// Set validity bit
	bitIdx := idx / 64
	bitPos := idx % 64
	if bitIdx < 2 {
		analysis.cacheF64Valid[bitIdx] |= 1 << bitPos
	}
}

/*
StatArchAnalysisGetCacheValue retrieves a cached value by StatKind from cacheOther.

Returns the value and true if cached, nil and false otherwise.
This method handles the index conversion (StatKind - 1) internally.
Used for non-float types (T, uint).

Time complexity: O(1)
Space complexity: O(1)
*/
func StatArchAnalysisGetCacheValue[T foundation.Numeric](analysis *StatArchAnalysis[T], kind StatKind) (interface{}, bool) {
	if kind < 1 || kind > StatKindCount {
		return nil, false
	}
	idx := kind - 1
	val := analysis.cacheOther[idx]
	return val, val != nil
}

/*
StatArchAnalysisSetCacheValue sets a cached value by StatKind in cacheOther.

This method handles the index conversion (StatKind - 1) internally.
Used for non-float types (T, uint).

Time complexity: O(1)
Space complexity: O(1)
*/
func StatArchAnalysisSetCacheValue[T foundation.Numeric](analysis *StatArchAnalysis[T], kind StatKind, value interface{}) {
	if kind < 1 || kind > StatKindCount {
		return
	}
	idx := kind - 1
	analysis.cacheOther[idx] = value
}

//go:linkname memclrHasPointers runtime.memclrHasPointers
//go:nosplit
func memclrHasPointers(ptr unsafe.Pointer, n uintptr)

//go:linkname memclrNoHeapPointers runtime.memclrNoHeapPointers
//go:nosplit
func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)

/*
WithMeanF32 sets the cached mean value in float32 precision.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithMeanF32(mean float32) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValueF32(a, StatKindMeanF32, mean)
	return a
}

/*
WithMeanF64 sets the cached mean value in float64 precision.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithMeanF64(mean float64) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValueF64(a, StatKindMeanF64, mean)
	return a
}

/*
WithSumF32 sets the cached sum value in float32 precision.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithSumF32(sum float32) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValueF32(a, StatKindSumF32, sum)
	return a
}

/*
WithSumF64 sets the cached sum value in float64 precision.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithSumF64(sum float64) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValueF64(a, StatKindSumF64, sum)
	return a
}

/*
WithNormSquaredF32 sets the cached norm squared (sum of squares) value in float32 precision.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithNormSquaredF32(normSquared float32) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValueF32(a, StatKindNormSquaredF32, normSquared)
	return a
}

/*
WithNormSquaredF64 sets the cached norm squared (sum of squares) value in float64 precision.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithNormSquaredF64(normSquared float64) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValueF64(a, StatKindNormSquaredF64, normSquared)
	return a
}

/*
WithVarianceF32 sets the cached variance value in float32 precision.

Parameters:
- variance: The variance value
- sample: If true, sets sample variance. If false, sets population variance.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithVarianceF32(variance float32, sample bool) *StatArchAnalysis[T] {
	if sample {
		StatArchAnalysisSetCacheValueF32(a, StatKindVarianceF32Sample, variance)
	} else {
		StatArchAnalysisSetCacheValueF32(a, StatKindVarianceF32Population, variance)
	}
	return a
}

/*
WithVarianceF64 sets the cached variance value in float64 precision.

Parameters:
- variance: The variance value
- sample: If true, sets sample variance. If false, sets population variance.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithVarianceF64(variance float64, sample bool) *StatArchAnalysis[T] {
	if sample {
		StatArchAnalysisSetCacheValueF64(a, StatKindVarianceF64Sample, variance)
	} else {
		StatArchAnalysisSetCacheValueF64(a, StatKindVarianceF64Population, variance)
	}
	return a
}

/*
WithStandardDeviationF32 sets the cached standard deviation value in float32 precision.

Parameters:
- stddev: The standard deviation value
- sample: If true, sets sample standard deviation. If false, sets population standard deviation.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithStandardDeviationF32(stddev float32, sample bool) *StatArchAnalysis[T] {
	if sample {
		StatArchAnalysisSetCacheValueF32(a, StatKindStddevF32Sample, stddev)
	} else {
		StatArchAnalysisSetCacheValueF32(a, StatKindStddevF32Population, stddev)
	}
	return a
}

/*
WithStandardDeviationF64 sets the cached standard deviation value in float64 precision.

Parameters:
- stddev: The standard deviation value
- sample: If true, sets sample standard deviation. If false, sets population standard deviation.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithStandardDeviationF64(stddev float64, sample bool) *StatArchAnalysis[T] {
	if sample {
		StatArchAnalysisSetCacheValueF64(a, StatKindStddevF64Sample, stddev)
	} else {
		StatArchAnalysisSetCacheValueF64(a, StatKindStddevF64Population, stddev)
	}
	return a
}

/*
WithMin sets the cached minimum value.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithMin(min T) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValue(a, StatKindMin, min)
	return a
}

/*
WithMax sets the cached maximum value.

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map
*/
func (a *StatArchAnalysis[T]) WithMax(max T) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValue(a, StatKindMax, max)
	return a
}

/*
WithCacheValue sets an arbitrary cached value by StatKind.

This method allows setting any cached statistic that may not have a dedicated
With method. Use this for less common statistics or when you need to set
multiple values of the same type.

Parameters:
- kind: The StatKind constant identifying the statistic
- value: The value to cache (must match the expected type for the StatKind)

Returns the analysis object for method chaining.

Time complexity: O(1) - map insertion is constant-time
Space complexity: O(1) - stores single value in existing cache map

Example:
```go
analysis.WithCacheValue(core.StatKindMedianF64, 42.5)
```
*/
func (a *StatArchAnalysis[T]) WithCacheValue(kind StatKind, value interface{}) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValue(a, kind, value)
	return a
}

func (a *StatArchAnalysis[T]) WithCacheValueF32(kind StatKind, value float32) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValueF32(a, kind, value)
	return a
}

func (a *StatArchAnalysis[T]) WithCacheValueF64(kind StatKind, value float64) *StatArchAnalysis[T] {
	StatArchAnalysisSetCacheValueF64(a, kind, value)
	return a
}

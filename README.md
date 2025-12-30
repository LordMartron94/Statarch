# Statarch

High-performance statistical computation library built on Blaze and manual memory management.

## Overview

`statarch` is a statistical computing library for Go that provides statistical operations, distributions, and analysis functions. It builds on top of `blaze` to leverage zero garbage collection overhead and predictable performance for statistical computations on large datasets.

## Design Philosophy

- **Zero GC Overhead**: All operations work on manually managed memory from the `memcore`/`memforge`/`memstruct`/`memarch` stack
- **Built on Blaze**: Uses `blaze` for underlying numeric computations (mean, sum, etc.)
- **Functional API**: Operations are stateless functions, not methods
- **Multiple Precisions**: Supports both `float32` and `float64` precision for operations
- **Statistical Focus**: Provides domain-specific statistical operations beyond basic numeric primitives
- **Computation Caching**: All computed statistics are cached automatically to avoid redundant calculations
- **Consistent API**: All functions operate on a single analysis structure for consistency and efficiency

## Performance Characteristics

- **Zero Allocations**: No heap allocations during computation
- **Cache Efficiency**: Uses contiguous memory from `memstruct.Vector` and `memstruct.Matrix`
- **Blaze Integration**: Leverages `blaze` for efficient numeric operations
- **Type Conversions**: F32/F64 variants may involve type conversions; use appropriate precision for your needs

## Integration

`statarch` is built on top of the manual memory management stack:

```text
memcore (foundation: memory mapping, marks, utilities)
    ↓
memforge (allocators) ──┐
    ↓                    │
memstruct (data structures) ──┐
    ↓                          │
memarch (factory layer: combines memforge + memstruct)
    ↓
blaze (uses memstruct.Vector and memstruct.Matrix)
    ↓
statarch (uses blaze for numeric operations)
```

**Dependency relationships:**

- **memcore**: Foundation providing memory mapping and mark system
- **memforge**: Allocators built on memcore (FixedLinearAllocator, etc.)
- **memstruct**: Data structures built on memcore (Vector, Matrix, Array, etc.)
- **memarch**: Factory layer that wraps memforge allocators to simplify creation of memstruct data structures
- **blaze**: Uses `memstruct.Vector` and `memstruct.Matrix` for numeric computations
- **statarch**: Uses `blaze` for underlying numeric operations (mean, sum, etc.)

**It requires:**

- `blaze` for underlying numeric operations (mean, sum, etc.)
- `memstruct.Vector` for vector operations
- `memstruct.Matrix` for matrix operations
- Properly initialized vectors/matrices (typically created using `memarch` factory functions)

## Packages

### `statarch/descriptive`

Descriptive statistics operations that summarize data characteristics.

**Analysis Structure:**

- `StatArchAnalysisCreate[T]` (from `statarch/core`) - Create an analysis structure for a vector
  - Requires an allocation function for creating internal sorted/scratch vectors on demand
  - All computed statistics are cached automatically
  - Sample and population statistics are cached separately for accuracy
  - Sorted and scratch vectors are created lazily and reused across operations
  - The analysis structure is shared across all StatArch packages (descriptive, hypothesis, etc.)

- `StatArchAnalysisBuilderCreate[T]` (from `statarch/core`) - Create a builder for pre-filling cached values
  - Allows clients to set pre-computed statistics (mean, sum, norm squared, variance, etc.)
  - Useful when statistics are already known from external computation
  - Provides fluent interface with method chaining
  - Example: `builder := core.StatArchAnalysisBuilderCreate[float64](vector, allocFn).WithMeanF64(42.5).WithSumF64(4250.0).Build()`

**Central Tendency:**

- `StatArchDescriptiveVectorMeanF32` / `StatArchDescriptiveVectorMeanF64` - Calculate arithmetic mean (cached)
- `StatArchDescriptiveVectorHarmonicMeanF32` / `StatArchDescriptiveVectorHarmonicMeanF64` - Calculate harmonic mean (cached, for rates/ratios)
- `StatArchDescriptiveVectorGeometricMeanF32` / `StatArchDescriptiveVectorGeometricMeanF64` - Calculate geometric mean (cached, for multiplicative processes)

**Pre-requisites (Cached for Reuse):**

- `StatArchDescriptiveVectorSumF32` / `StatArchDescriptiveVectorSumF64` - Calculate vector sum (cached, used by distance functions)
- `StatArchDescriptiveVectorNormSquaredF32` / `StatArchDescriptiveVectorNormSquaredF64` - Calculate sum of squares / norm squared (cached, used by cosine similarity)
- `StatArchDescriptiveVectorTrimmedMeanF32` / `StatArchDescriptiveVectorTrimmedMeanF64` - Calculate trimmed mean (robust to outliers, not cached due to continuous trim parameter)
- `StatArchDescriptiveVectorMedianF32` / `StatArchDescriptiveVectorMedianF64` - Calculate median (cached, handles sorting internally)
- `StatArchDescriptiveVectorMode` - Calculate mode (cached, handles sorting internally)

**Percentiles and Quantiles:**

- `StatArchDescriptiveVectorPercentileF32` / `StatArchDescriptiveVectorPercentileF64` - Calculate percentile (handles sorting internally, not cached due to large number of possible values)
- `StatArchDescriptiveVectorIQRF32` / `StatArchDescriptiveVectorIQRF64` - Calculate interquartile range (cached, handles sorting internally)
- `StatArchDescriptiveVectorNormalizedIQRF32` / `StatArchDescriptiveVectorNormalizedIQRF64` - Calculate normalized IQR (cached, handles sorting internally)

**Spread and Variability:**

- `StatArchDescriptiveVectorVarianceF32` / `StatArchDescriptiveVectorVarianceF64` - Calculate variance (sample or population, cached separately)
  - **Numerical Stability Note:** Uses the naive variance formula (E[X²] - E[X]²) which can suffer from numerical instability for large values or when the mean is much smaller than the variance. For maximum numerical stability, use `StatArchDescriptiveVectorAnalyzeWelford`. See "Accuracy and Simplifications" section for details.
- `StatArchDescriptiveVectorStandardDeviationF32` / `StatArchDescriptiveVectorStandardDeviationF64` - Calculate standard deviation (sample or population, cached separately, uses cached variance)
- `StatArchDescriptiveVectorRange` - Calculate range (max - min, cached)
  - **Note:** Range is sensitive to outliers. Consider using IQR for more robust spread measurement.
- `StatArchDescriptiveVectorMeanAbsoluteDeviationF32` / `StatArchDescriptiveVectorMeanAbsoluteDeviationF64` - Calculate mean absolute deviation (cached, uses cached mean)
- `StatArchDescriptiveVectorMedianAbsoluteDeviationF32` / `StatArchDescriptiveVectorMedianAbsoluteDeviationF64` - Calculate median absolute deviation (cached, handles sorting and scratch vectors internally)
  - **Note:** The function computes median of absolute deviations from the median, then scales by 1.4826 to make it comparable to standard deviation for normal distributions.

**Distribution Shape:**

- `StatArchDescriptiveVectorSkewnessF32` / `StatArchDescriptiveVectorSkewnessF64` - Calculate skewness (sample or population, cached separately)
- `StatArchDescriptiveVectorKurtosisF32` / `StatArchDescriptiveVectorKurtosisF64` - Calculate kurtosis (sample or population, cached separately)
- `StatArchDescriptiveVectorAnalyzeWelford` - Compute mean, variance, skewness, and kurtosis in a single pass using Welford's online algorithm (numerically stable, recommended for large datasets or high-precision requirements, updates cache)

**Normalized Measures:**

- `StatArchDescriptiveVectorCoefficientVariantF32` / `StatArchDescriptiveVectorCoefficientVariantF64` - Calculate coefficient of variation (sample or population, cached separately)
- `StatArchDescriptiveVectorStandardErrorMeanF32` / `StatArchDescriptiveVectorStandardErrorMeanF64` - Calculate standard error of the mean (sample or population, cached separately)

**Robust Measures:**

- `StatArchDescriptiveVectorWinsorizedMeanF32` / `StatArchDescriptiveVectorWinsorizedMeanF64` - Calculate winsorized mean (robust to outliers, not cached due to continuous trim parameter)
- `StatArchDescriptiveVectorMidRangeF32` / `StatArchDescriptiveVectorMidRangeF64` - Calculate mid-range (average of min and max, cached)
- `StatArchDescriptiveVectorFiveNumberSummaryF32` / `StatArchDescriptiveVectorFiveNumberSummaryF64` - Calculate five-number summary (Min, Q1, Median, Q3, Max) in a single operation
- `StatArchDescriptiveVectorBimodalityCoefficientF32` / `StatArchDescriptiveVectorBimodalityCoefficientF64` - Calculate bimodality coefficient (values > 0.555 suggest bimodality, cached)

**Advanced Quantiles:**

- `StatArchDescriptiveVectorQuantileBierensF32` / `StatArchDescriptiveVectorQuantileBierensF64` - Bierens' kernel-based quantile estimator (robust for small samples, not cached due to large number of possible percentile values)
  - **Accuracy Note:** Bandwidth selection uses a simplified adaptive approach. For research-grade accuracy with specific distribution assumptions, consider more sophisticated bandwidth selection. See "Accuracy and Simplifications" section for details.
- `StatArchDescriptiveVectorQuantileHarrellDavisF32` / `StatArchDescriptiveVectorQuantileHarrellDavisF64` - Harrell-Davis beta distribution-based quantile estimator (not cached due to large number of possible percentile values)
  - **Accuracy Note:** This implementation uses the regularized incomplete beta function from `blaze/math`, which provides high accuracy even in the tails (P99.9) for risk analysis and SLA monitoring.
- `StatArchDescriptiveVectorQuantileRangeF32` / `StatArchDescriptiveVectorQuantileRangeF64` - Generic quantile ratio (e.g., P99/P50 for long tail measurement, not cached due to large number of possible percentile combinations)
- `StatArchDescriptiveVectorDecileRatioF32` / `StatArchDescriptiveVectorDecileRatioF64` - Decile ratio (P90/P10, commonly used in economics, not cached - computed from percentiles which are not cached)

**Extreme Values:**

- `StatArchDescriptiveVectorTopK` - Extract top K largest values (sorted descending, not cached - parameter-dependent, returns values not statistics)
- `StatArchDescriptiveVectorBottomK` - Extract bottom K smallest values (sorted ascending, not cached - parameter-dependent, returns values not statistics)

**Vector Transformations:**

- `StatArchDescriptiveVectorZScoreNormalizeF32` / `StatArchDescriptiveVectorZScoreNormalizeF64` - Normalize vector to z-scores (standardization: mean=0, stddev=1)

**Example:**

```go
import (
    "memcore"
    "memforge"
    "memarch"
    "statarch/descriptive"
)

// Create allocator and vector
allocator := memforge.FixedLinearAllocatorCreate(uint64(memcore.MegaByte))
defer memforge.FixedLinearAllocatorDestroy(allocator)

allocFn := func(sizeBytes, alignment uint64) memcore.MarkRaw {
    return memforge.FixedLinearAllocatorMalloc(allocator, sizeBytes, alignment)
}

vectorMark, _ := memarch.MemArchVectorCreate[float64](allocFn, 1000)
// ... populate vector ...

// Create analysis structure (requires allocFn for internal sorted/scratch vectors)
import "statarch/core"
analysis := core.StatArchAnalysisCreate[float64](vectorMark, allocFn)

// Alternatively, use builder pattern to pre-fill cached values if already known
// This is useful when you've computed statistics externally and want to avoid recomputation
builder := core.StatArchAnalysisBuilderCreate[float64](vectorMark, allocFn)
analysisWithCache := builder.
    WithMeanF64(42.5).
    WithSumF64(4250.0).
    WithNormSquaredF64(100000.0).
    Build()

// All functions operate on the analysis structure and cache their results
// First call computes and caches, subsequent calls return cached values
mean := descriptive.StatArchDescriptiveVectorMeanF64(analysis)
harmonicMean := descriptive.StatArchDescriptiveVectorHarmonicMeanF64(analysis) // for rates/ratios
geometricMean := descriptive.StatArchDescriptiveVectorGeometricMeanF64(analysis) // for multiplicative processes
trimmedMean := descriptive.StatArchDescriptiveVectorTrimmedMeanF64(analysis, 0.1) // trim 10% from each end
variance := descriptive.StatArchDescriptiveVectorVarianceF64(analysis, true) // sample variance (cached separately from population)
stddev := descriptive.StatArchDescriptiveVectorStandardDeviationF64(analysis, true) // uses cached variance
sem := descriptive.StatArchDescriptiveVectorStandardErrorMeanF64(analysis, true) // standard error of mean
skewness := descriptive.StatArchDescriptiveVectorSkewnessF64(analysis, true) // uses cached mean and stddev
median := descriptive.StatArchDescriptiveVectorMedianF64(analysis) // creates sorted copy on first call, then cached
mode, occurrence := descriptive.StatArchDescriptiveVectorMode(analysis) // creates scratch vector on first call, then cached
percentile := descriptive.StatArchDescriptiveVectorPercentileF64(analysis, 95) // not cached (too many possible values)
iqr := descriptive.StatArchDescriptiveVectorIQRF64(analysis) // cached, reuses sorted vector
normalizedIQR := descriptive.StatArchDescriptiveVectorNormalizedIQRF64(analysis) // cached, relative measure
winsorizedMean := descriptive.StatArchDescriptiveVectorWinsorizedMeanF64(analysis, 0.1) // winsorize 10% from each end
midRange := descriptive.StatArchDescriptiveVectorMidRangeF64(analysis) // cached
fiveNum := descriptive.StatArchDescriptiveVectorFiveNumberSummaryF64(analysis) // Min, Q1, Median, Q3, Max
bimodality := descriptive.StatArchDescriptiveVectorBimodalityCoefficientF64(analysis, true) // test for bimodality
quantileBierens := descriptive.StatArchDescriptiveVectorQuantileBierensF64(analysis, 95) // robust quantile
quantileHD := descriptive.StatArchDescriptiveVectorQuantileHarrellDavisF64(analysis, 95) // Harrell-Davis quantile
quantileRatio := descriptive.StatArchDescriptiveVectorQuantileRangeF64(analysis, 99, 50) // P99/P50 ratio
decileRatio := descriptive.StatArchDescriptiveVectorDecileRatioF64(analysis) // P90/P10 ratio
descriptive.StatArchDescriptiveVectorTopK(analysis, 10, top10Vector) // extract top 10 values
descriptive.StatArchDescriptiveVectorBottomK(analysis, 10, bottom10Vector) // extract bottom 10 values

// Vector transformations
zScoreVectorMark, _ := memarch.MemArchVectorCreate[float64](allocFn, 1000)
descriptive.StatArchDescriptiveVectorZScoreNormalizeF64(analysis, true, zScoreVectorMark) // normalizes to z-scores
```

## Use Cases

- **Data Analysis**: Descriptive statistics, correlation analysis, distribution fitting
- **Scientific Computing**: Statistical modeling, hypothesis testing, experimental analysis
- **Machine Learning**: Feature analysis, data preprocessing, model evaluation metrics
- **Financial Computing**: Risk analysis, portfolio statistics, time series analysis
- **Quality Control**: Process monitoring, statistical process control, outlier detection

## Safety Guidelines

⚠️ **Important:**

1. **Input Validation**: Ensure vectors/matrices have compatible dimensions for operations
2. **Memory Lifetime**: Input and output vectors/matrices must be valid during operations
3. **Type Consistency**: Use consistent numeric types (mixing may cause unexpected behavior)
4. **Precision**: Be aware of precision loss when using F32 variants with large numbers
5. **Sample Size**: Some statistical operations require minimum sample sizes for validity
6. **Allocation Function**: The analysis structure requires a valid allocation function for creating internal sorted/scratch vectors
7. **Error Handling**: Functions panic on invalid inputs (e.g., variance with < 2 items, percentile > 100). Check prerequisites before calling.
8. **Caching**: All computed statistics are cached automatically. Sample and population variants are cached separately for accuracy.
9. **Cache Invalidation**: The analysis structure automatically detects when the underlying vector is modified and invalidates cached statistics. This prevents stale data issues in long-running processes or streaming scenarios. Version tracking is transparent - you can modify vectors freely and cached values will be recomputed as needed.

**Error Handling Behavior:**

- **Panics**: Used for invalid inputs or mathematically undefined operations
  - Variance/Stddev with < 2 items
  - Skewness with < 2 items (population) or < 3 items (sample)
  - Kurtosis with < 3 items (population) or < 4 items (sample)
  - Kurtosis with standard deviation = 0
  - Coefficient of variation with mean = 0
  - Percentile > 100
  - Harmonic mean with zero values
  - Geometric mean with zero or negative values
  - Trimmed mean with trimPercent outside [0.0, 0.5]
  - Z-score normalization with standard deviation = 0 or destination capacity mismatch

- **Returns Zero Values**: Used for empty vectors or valid edge cases
  - Mean, median, percentile return 0 for empty vectors
  - Mode returns zero value and 0 occurrence for empty vectors
  - Range, IQR, MAD return 0 for empty vectors
  - Skewness returns 0 if standard deviation is 0

## Implementation Notes

This section documents implementation details and positive characteristics of specific algorithms.

### Harrell-Davis Quantile Estimator

**Functions:** `StatArchDescriptiveVectorQuantileHarrellDavisF32` / `StatArchDescriptiveVectorQuantileHarrellDavisF64`

**Implementation:** Uses the regularized incomplete beta function from `blaze/math`, which implements the Lentz continued fraction method. This provides high accuracy even in the tails (P99.9) for risk analysis and SLA monitoring.

**When to use:** Suitable for all quantile estimation use cases, including critical applications requiring accurate tail quantiles.

### Spearman Correlation Ranking

**Functions:** `StatArchCorrelationVectorSpearmanF32` / `StatArchCorrelationVectorSpearmanF64`

**Implementation:** Ranking uses a high-performance approach with O(n log n) time complexity. The implementation creates a temporary `rankEntry` structure that binds each value with its original index. A single `Array` of these entries is sorted once using quicksort, then ranks are assigned linearly by iterating through the sorted entries. Ties are handled by calculating average ranks for contiguous blocks of equal values.

**Performance:** Optimal O(n log n) performance using iterative quicksort with median-of-three pivot selection. The algorithm correctly handles ties by assigning average ranks.

**Note:** This function accepts raw vectors (not `StatArchAnalysis` structures) because ranking requires creating scratch vectors. The function internally creates analysis structures for the ranked vectors before computing Pearson correlation.

## Accuracy and Limitations

⚠️ **Important:** This section documents known limitations and accuracy considerations for specific operations.

### Variance Calculations

**Functions:** Individual variance functions (e.g., `StatArchDescriptiveVectorVarianceF32/F64`)

**Limitation:** Some variance calculations use the naive formula (E[X²] - E[X]²) which can suffer from numerical instability for large values or when the mean is much smaller than the variance.

**Impact:** For datasets with large values or high variance-to-mean ratios, floating-point precision errors may accumulate. The naive formula can lose precision due to catastrophic cancellation.

**Recommendation:** For maximum numerical stability, use `StatArchDescriptiveVectorAnalyzeWelford` which uses Welford's online algorithm. This algorithm is numerically stable and computes mean, variance, skewness, and kurtosis in a single pass with superior numerical properties.

**When to use:** 
- Individual variance functions: Suitable for most practical purposes with reasonable value ranges
- Welford's algorithm: Recommended for large datasets, high-precision requirements, or when computing multiple statistics together

### Bierens Quantile Estimator

**Functions:** `StatArchDescriptiveVectorQuantileBierensF32` / `StatArchDescriptiveVectorQuantileBierensF64`

**Limitation:** Bandwidth selection uses a simplified adaptive approach. More sophisticated bandwidth selection methods (e.g., cross-validation, plug-in estimators) could improve accuracy for certain data distributions.

**Impact:** The Epanechnikov kernel implementation is standard, but bandwidth selection may not be optimal for all data types.

**When to use:** Suitable for robust quantile estimation, especially for small samples. For research-grade accuracy with specific distribution assumptions, consider more sophisticated bandwidth selection.

### `statarch/correlation`

Correlation and relationship measures between pairs of vectors, and autocorrelation for time series analysis.

All functions accept `StatArchAnalysis` structures and use cached values (means, standard deviations) when available to avoid redundant computations. The correlation results themselves are not cached since they are computed between vectors or at different lags.

**Functions:**

- `StatArchCorrelationVectorCovarianceF32` / `StatArchCorrelationVectorCovarianceF64` - Compute covariance (sample or population)
  - Uses cached means from analysis structures when available
  - Formula: cov(X,Y) = E[(X - μX)(Y - μY)] with sample/population correction
  
- `StatArchCorrelationVectorPearsonF32` / `StatArchCorrelationVectorPearsonF64` - Compute Pearson correlation coefficient (linear relationship, -1 to +1)
  - Uses cached means and standard deviations from analysis structures when available
  - Formula: r = cov(X,Y) / (σX * σY)
  
- `StatArchCorrelationVectorSpearmanF32` / `StatArchCorrelationVectorSpearmanF64` - Compute Spearman rank correlation (monotonic relationship, -1 to +1)
  - Accepts analysis structures and uses their allocFn for creating scratch vectors for ranking
  - Handles ties by assigning average ranks
  - **Performance:** Uses O(n log n) quicksort with a single Array of rankEntry structs that bind values with their original indices. Optimal performance for all dataset sizes.
  
- `StatArchCorrelationVectorCosineSimilarityF32` / `StatArchCorrelationVectorCosineSimilarityF64` - Compute cosine similarity (directional similarity, -1 to +1)
  - Accepts analysis structures and uses cached norm squared values if available
  - Formula: cos(θ) = (A · B) / (||A|| * ||B||)
  - Norm squared values are automatically cached in the analysis structure for reuse
  
- `StatArchCorrelationVectorAutocorrelationF32` / `StatArchCorrelationVectorAutocorrelationF64` - Compute autocorrelation at a given lag (serial correlation, -1 to +1)
  - Accepts analysis structure and lag parameter
  - Measures correlation of a signal with a delayed copy of itself
  - Returns 1.0 at lag 0 (perfect correlation with itself)
  - Formula: r(k) = corr(X[0:n-k], X[k:n]) where k is the lag
  - Use cases: time series analysis, signal processing, detecting periodicity, randomness testing

**Example:**

```go
import (
    "statarch/correlation"
    "statarch/core"
)

// Create analysis structures for vectors
analysisA := core.StatArchAnalysisCreate[float64](vectorA, allocFn)
analysisB := core.StatArchAnalysisCreate[float64](vectorB, allocFn)

// Compute covariance (uses cached means if available)
covariance := correlation.StatArchCorrelationVectorCovarianceF64(analysisA, analysisB, true)

// Compute Pearson correlation (uses cached means and stddev if available)
pearson := correlation.StatArchCorrelationVectorPearsonF64(analysisA, analysisB, true)

// Compute Spearman correlation (uses analysis structures, allocFn from analysis)
spearman := correlation.StatArchCorrelationVectorSpearmanF64(analysisA, analysisB)

// Compute cosine similarity (uses analysis structures, norms are cached)
cosine := correlation.StatArchCorrelationVectorCosineSimilarityF64(analysisA, analysisB)

// Compute autocorrelation at lag 1 (correlation with itself shifted by 1 position)
autocorrLag1 := correlation.StatArchCorrelationVectorAutocorrelationF64(analysisA, 1, true)
// Compute autocorrelation at lag 5
autocorrLag5 := correlation.StatArchCorrelationVectorAutocorrelationF64(analysisA, 5, true)
```

### `statarch/distance`

Distance and divergence measures between probability distributions.

All functions accept `StatArchAnalysis` structures and use cached sums if available, avoiding redundant computations. The distance results themselves are not cached since they are computed between two vectors.

**Functions:**

- `StatArchDistanceVectorBhattacharyyaF32` / `StatArchDistanceVectorBhattacharyyaF64` - Compute Bhattacharyya distance (distribution similarity, 0 to ∞)
  - Measures similarity between two probability distributions
  - Based on Bhattacharyya coefficient (overlap between distributions)
  - Formula: D_B = -ln(Σ√(p_i * q_i)) where p, q are normalized probability vectors
  - Requires non-negative values (probabilities)
  - Range: 0 to ∞ (0 = identical distributions)
  
- `StatArchDistanceVectorKLDivergenceF32` / `StatArchDistanceVectorKLDivergenceF64` - Compute Kullback-Leibler divergence (asymmetric, 0 to ∞)
  - Measures how much one probability distribution diverges from another
  - Asymmetric: KL(P||Q) ≠ KL(Q||P), where P is the "true" distribution and Q is the approximation
  - Formula: KL(P||Q) = Σ p_i * ln(p_i / q_i)
  - Skips terms where p_i = 0 (0 * log(0/q) = 0)
  - Panics if q_i = 0 and p_i > 0 (undefined: log(p/0))
  - Requires non-negative values
  - Range: 0 to ∞ (0 = identical distributions)

**Example:**

```go
import (
    "statarch/distance"
    "statarch/core"
)

// Create analysis structures for probability distributions
analysisA := core.StatArchAnalysisCreate[float64](distributionA, allocFn)
analysisB := core.StatArchAnalysisCreate[float64](distributionB, allocFn)

// Compute Bhattacharyya distance (for detecting distribution drift)
bhattacharyya := distance.StatArchDistanceVectorBhattacharyyaF64(analysisA, analysisB)

// Compute KL divergence (P is true distribution, Q is approximation)
analysisP := core.StatArchAnalysisCreate[float64](vectorP, allocFn)
analysisQ := core.StatArchAnalysisCreate[float64](vectorQ, allocFn)
klDivergence := distance.StatArchDistanceVectorKLDivergenceF64(analysisP, analysisQ)
```

### `statarch/hypothesis`

Statistical hypothesis tests.

All functions accept `StatArchAnalysis` structures and use cached values (skewness, kurtosis) when available.

**Functions:**

- `StatArchHypothesisVectorJarqueBeraF32` / `StatArchHypothesisVectorJarqueBeraF64` - Jarque-Bera normality test (chi-squared with 2 degrees of freedom)
  - Goodness-of-fit test that determines if sample data has skewness and kurtosis matching a normal distribution
  - Uses cached skewness and kurtosis from analysis structure
  - Test statistic follows a chi-squared distribution with 2 degrees of freedom
  - Formula: JB = (n/6) * (skewness² + (kurtosis²/4))
  - Large values indicate deviation from normality
  - Critical values: 5.99 (α=0.05), 9.21 (α=0.01)
  - Requires at least 3 elements (population) or 4 elements (sample) for skewness/kurtosis

- `StatArchHypothesisVectorMannWhitneyUF32` / `StatArchHypothesisVectorMannWhitneyUF64` - Mann-Whitney U test (non-parametric alternative to t-test)
  - Non-parametric test that determines whether two independent samples come from the same distribution
  - Standard alternative to the t-test when data is not normally distributed
  - Returns `MannWhitneyResult` containing U statistic, z-score, and two-tailed p-value
  - For large samples (n_A + n_B > 20): Uses normal approximation with tie correction
  - For small samples (n_A + n_B ≤ 20): Returns U statistic with NaN for z-score and p-value (exact distribution computation would be needed)
  - Handles ties by adjusting variance using correction factor
  - Time complexity: O(n log n) - dominated by ranking operation
  - Requires `allocFn` for creating temporary vectors for ranking

**Example:**

```go
import (
    "statarch/hypothesis"
    "statarch/core"
)

// Create analysis structure
analysis := core.StatArchAnalysisCreate[float64](vectorMark, allocFn)

// Test for normality using Jarque-Bera test (uses cached skewness and kurtosis)
jbStatistic := hypothesis.StatArchHypothesisVectorJarqueBeraF64(analysis, true)
// Compare against critical values: 5.99 (α=0.05), 9.21 (α=0.01)

// If normality test fails, use Mann-Whitney U test for comparing two groups
analysisA := core.StatArchAnalysisCreate[float64](groupAVector, allocFn)
analysisB := core.StatArchAnalysisCreate[float64](groupBVector, allocFn)
result := hypothesis.StatArchHypothesisVectorMannWhitneyUF64(analysisA, analysisB, allocFn)
// result.UStatistic - the U statistic
// result.ZScore - z-score for normal approximation (only valid for large samples)
// result.PValue - two-tailed p-value (only valid for large samples)
```

### `statarch/multivariate`

Matrix-based multivariate statistical operations for datasets with multiple features.

This package extends statarch's vector-based operations to handle matrices of features, enabling efficient computation of correlation and covariance matrices for datasets with multiple variables. All operations leverage Blaze's matrix multiplication capabilities for optimal performance, providing massive speedups over nested loops of pairwise vector operations.

**Functions:**

- `StatArchMultivariateCovarianceMatrixF32` / `StatArchMultivariateCovarianceMatrixF64` - Compute covariance matrix for a data matrix
  - Input: Matrix (n samples × p features), sample flag
  - Output: Covariance matrix (p × p)
  - Algorithm: Mean-center matrix, compute X^T X using Blaze matrix multiplication, divide by (n-1) for sample or n for population
  - Time complexity: O(n*p²) - dominated by matrix multiplication
  - Leverages Blaze's optimized matrix operations for massive speedups over pairwise vector operations
  - Requires `allocFn` for creating temporary matrices

- `StatArchMultivariateCorrelationMatrixF32` / `StatArchMultivariateCorrelationMatrixF64` - Compute correlation matrix for a data matrix
  - Input: Matrix (n samples × p features), sample flag
  - Output: Correlation matrix (p × p)
  - Algorithm: Compute covariance matrix, extract diagonal (variances), normalize by standard deviations
  - Diagonal elements are always 1.0 (perfect correlation with itself)
  - Panics if any column has zero variance (cannot compute correlation)
  - Time complexity: O(n*p²) - dominated by covariance matrix computation
  - Requires `allocFn` for creating temporary matrices

**Example:**

```go
import (
    "memarch"
    "statarch/multivariate"
)

// Create data matrix: 1000 samples × 100 features
dataMatrix, _ := memarch.MemArchMatrixCreate[float64](allocFn, 1000, 100)
// ... populate matrix ...

// Create output covariance matrix: 100 × 100
covMatrix, _ := memarch.MemArchMatrixCreate[float64](allocFn, 100, 100)

// Compute covariance matrix (sample)
multivariate.StatArchMultivariateCovarianceMatrixF64(dataMatrix, covMatrix, true, allocFn)

// Create output correlation matrix: 100 × 100
corrMatrix, _ := memarch.MemArchMatrixCreate[float64](allocFn, 100, 100)

// Compute correlation matrix (sample)
multivariate.StatArchMultivariateCorrelationMatrixF64(dataMatrix, corrMatrix, true, allocFn)
```

**Use Cases:**

- Feature analysis in machine learning (understanding feature relationships)
- Financial analysis (portfolio risk, asset correlations)
- Multivariate statistical analysis
- Principal Component Analysis (PCA) preprocessing

**Performance:**

- Leverages Blaze's matrix multiplication (X^T X) for optimal performance
- Provides massive speedups over nested loops of pairwise vector operations
- For a 100×100 correlation matrix, avoids 4,950 pairwise function calls

### `statarch/distribution`

Statistical distribution functions.

This package wraps Blaze's mathematical primitives (Gamma, Beta functions) to provide domain-specific statistical distribution operations such as CDF, PDF, and quantile functions.

**Functions:**

- `StatArchDistributionBetaCDFF32` / `StatArchDistributionBetaCDFF64` - Beta distribution cumulative distribution function (CDF)
  - Input: x (0 ≤ x ≤ 1), alpha, beta parameters
  - Output: CDF value P(X ≤ x)
  - Direct call to Blaze's optimized beta function (Lentz continued fraction method)
  - Formula: CDF(x) = I_x(α, β) where I_x is the regularized incomplete beta function
  - High accuracy even in the tails

- `StatArchDistributionBetaPDFF32` / `StatArchDistributionBetaPDFF64` - Beta distribution probability density function (PDF)
  - Input: x (0 < x < 1), alpha, beta parameters
  - Output: PDF value
  - Uses log-space computation for numerical stability
  - Formula: PDF(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)
  - Handles boundary cases (x = 0, x = 1) based on alpha and beta values

- `StatArchDistributionBetaQuantileF32` / `StatArchDistributionBetaQuantileF64` - Beta distribution quantile (inverse CDF)
  - Input: p (0 ≤ p ≤ 1), alpha, beta parameters
  - Output: Quantile value x such that CDF(x) = p
  - Uses Newton-Raphson iteration with binary search fallback
  - Typically converges in 3-5 iterations
  - Handles edge cases: p = 0 → returns 0, p = 1 → returns 1

**Example:**

```go
import "statarch/distribution"

// Compute Beta CDF
alpha := 2.0
beta := 5.0
x := 0.3
cdf := distribution.StatArchDistributionBetaCDFF64(x, alpha, beta)
// cdf = P(X ≤ 0.3) for Beta(2, 5) distribution

// Compute Beta PDF
pdf := distribution.StatArchDistributionBetaPDFF64(x, alpha, beta)
// pdf = probability density at x = 0.3

// Compute Beta quantile (inverse CDF)
p := 0.95
quantile := distribution.StatArchDistributionBetaQuantileF64(p, alpha, beta)
// quantile = value such that P(X ≤ quantile) = 0.95
```

**Use Cases:**

- Statistical hypothesis testing
- Bayesian inference
- Confidence interval construction
- Random variate generation (inverse transform sampling)
- Statistical modeling

**Implementation:**

- CDF: Direct call to `blaze/math.BlazeMathBetaRegularizedIncompleteF64` (Lentz continued fraction method)
- PDF: Log-space computation using `BlazeMathLogGammaAbs` for numerical stability
- Quantile: Newton-Raphson iteration with binary search fallback for robustness

### `statarch/rolling`

Rolling statistics (moving windows) for time-series analysis and real-time monitoring.

All functions accept `StatArchRollingAnalysis` structures and use cached values when available. Rolling statistics use Welford's online algorithm for numerically stable mean and variance computation, maintaining state that updates incrementally as new values are added to the window.

**Analysis Structure:**

- `StatArchRollingAnalysisCreate[T]` (from `statarch/rolling`) - Create an analysis structure for a circular buffer
  - Requires an allocation function for creating internal scratch vectors on demand
  - All computed statistics are cached automatically
  - Sample and population statistics are cached separately for accuracy
  - Welford's algorithm state (n, M1, M2, M3, M4) is maintained incrementally for mean, variance, skewness, and kurtosis
  - The analysis structure automatically detects when the underlying buffer is modified and invalidates cached statistics

**Central Tendency:**

- `StatArchRollingVectorMeanF32` / `StatArchRollingVectorMeanF64` - Calculate rolling mean using Welford's algorithm (cached)
- `StatArchRollingVectorSum` - Calculate rolling sum (cached, maintains running sum)

**Spread and Variability:**

- `StatArchRollingVectorVarianceF32` / `StatArchRollingVectorVarianceF64` - Calculate rolling variance using Welford's algorithm (cached, sample or population)
  - **Numerical Stability Note:** Uses Welford's online algorithm for superior numerical stability, especially important for rolling statistics where values are constantly being added and removed
- `StatArchRollingVectorStandardDeviationF32` / `StatArchRollingVectorStandardDeviationF64` - Calculate rolling standard deviation (cached, sample or population, uses cached variance)

**Extreme Values:**

- `StatArchRollingVectorMin` - Calculate rolling minimum (cached, maintains running min)
- `StatArchRollingVectorMax` - Calculate rolling maximum (cached, maintains running max)

**Example:**

```go
import (
    "memcore"
    "memforge"
    "memarch"
    "statarch/rolling"
)

// Create allocator and circular buffer
allocator := memforge.FixedLinearAllocatorCreate(uint64(memcore.MegaByte))
defer memforge.FixedLinearAllocatorDestroy(allocator)

allocFn := func(sizeBytes, alignment uint64) memcore.MarkRaw {
    return memforge.FixedLinearAllocatorMalloc(allocator, sizeBytes, alignment)
}

// Create circular buffer with window size of 100
bufferMark, _ := memarch.MemArchCircularBufferCreate[float64](allocFn, 100)

// Create analysis structure
analysis := rolling.StatArchRollingAnalysisCreate[float64](bufferMark, allocFn)

// Add values to the buffer (automatically removes oldest when full)
memstruct.CircularBufferPush[float64](bufferMark, 10.5)
memstruct.CircularBufferPush[float64](bufferMark, 11.2)
memstruct.CircularBufferPush[float64](bufferMark, 9.8)
// ... continue adding values ...

// All functions operate on the analysis structure and cache their results
// First call computes and caches, subsequent calls return cached values (if buffer hasn't changed)
mean := rolling.StatArchRollingVectorMeanF64(analysis)
variance := rolling.StatArchRollingVectorVarianceF64(analysis, true) // sample variance
stddev := rolling.StatArchRollingVectorStandardDeviationF64(analysis, true) // uses cached variance
min := rolling.StatArchRollingVectorMin[float64](analysis)
max := rolling.StatArchRollingVectorMax[float64](analysis)
sum := rolling.StatArchRollingVectorSum[float64](analysis)
skewness := rolling.StatArchRollingVectorSkewnessF64(analysis, true) // sample skewness
kurtosis := rolling.StatArchRollingVectorKurtosisF64(analysis, true) // sample kurtosis

// As you add more values, the window slides and statistics update automatically
// The cache invalidates when the buffer version changes (on every push)

// Exponentially Weighted Moving Average (infinite window, zero memory growth)
ewma := rolling.StatArchRollingExponentialMovingAverageCreate[float64](0.3) // alpha = 0.3 for smoothing
rolling.StatArchRollingExponentialMovingAverageUpdate(ewma, 10.5)
rolling.StatArchRollingExponentialMovingAverageUpdate(ewma, 11.2)
rolling.StatArchRollingExponentialMovingAverageUpdate(ewma, 9.8)
ewmaValue := rolling.StatArchRollingExponentialMovingAverageGetF64(ewma) // current EWMA value
```

**Use Cases:**

- **Real-time Anomaly Detection**: Monitor rolling mean and variance to detect outliers
- **Signal Processing**: Track rolling statistics of time-series signals
- **Financial Computing**: Monitor rolling volatility, moving averages, price ranges
- **Process Monitoring**: Track rolling statistics of system metrics

**Performance Characteristics:**

- **O(1) Updates**: Welford's algorithm maintains state incrementally, avoiding full recomputation
- **Cache Efficiency**: All statistics are cached and invalidated automatically on buffer modifications
- **Zero Allocations**: No heap allocations during computation
- **Version Tracking**: Automatic cache invalidation when buffer is modified

**Integration:**

Rolling statistics require a `CircularBuffer` from `memstruct`, which can be created using `memarch.MemArchCircularBufferCreate`. The buffer maintains a fixed-size sliding window - when full, adding a new value automatically removes the oldest value.

## Relationship to Blaze

`statarch` provides **statistical domain operations** that build on `blaze`'s **numeric primitives**:

- **Blaze**: Provides basic numeric operations (sum, mean, elementwise operations, etc.)
- **Statarch**: Provides statistical operations (variance, correlation, distributions, hypothesis tests, etc.)

This separation follows the architectural principle of building focused, reusable capabilities. Use `blaze` for numeric computation primitives, and `statarch` for statistical analysis.

## Future Directions

`statarch` is designed for extensibility. Potential future additions:

- More distribution types (exponential, gamma, normal, chi-squared, etc.)
- Additional rolling statistics (rolling median, rolling skewness/kurtosis, exponential weighted moving statistics)
- Additional non-parametric tests (Kruskal-Wallis, Wilcoxon signed-rank, etc.)
- Bayesian statistics operations
- Additional multivariate statistics (principal component analysis, factor analysis, etc.)

## Why Statarch?

Traditional Go statistical libraries often:

- Trigger garbage collection during computation
- Have unpredictable performance characteristics
- Don't integrate with manual memory management
- Require copying data between different representations

`statarch` provides:

- Zero GC overhead
- Predictable, cache-friendly performance
- Explicit precision control
- Direct operation on manual memory structures
- Statistical operations built on proven numeric primitives (`blaze`)

Use `statarch` when you need high-performance statistical computations with deterministic memory usage.
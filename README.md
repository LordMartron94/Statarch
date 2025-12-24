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

## Relationship to Blaze

`statarch` provides **statistical domain operations** that build on `blaze`'s **numeric primitives**:

- **Blaze**: Provides basic numeric operations (sum, mean, elementwise operations, etc.)
- **Statarch**: Provides statistical operations (variance, correlation, distributions, hypothesis tests, etc.)

This separation follows the architectural principle of building focused, reusable capabilities. Use `blaze` for numeric computation primitives, and `statarch` for statistical analysis.

## Future Directions

`statarch` is designed for extensibility. Potential future additions:

- More distribution types (exponential, gamma, beta, etc.)
- Time series analysis operations
- Non-parametric tests (Mann-Whitney, Kruskal-Wallis, etc.)
- Bayesian statistics operations
- Multivariate statistics operations

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

package multivariate

import (
	"blaze/scalar"
	"blaze/structure"
	"foundation"
	"memarch"
	"memcore"
	"memstruct"
)

/*
StatArchMultivariateCovarianceMatrixF32 computes the covariance matrix for a data matrix in float32 precision.

The input matrix X has dimensions n×p where:
- n is the number of samples (rows)
- p is the number of features (columns)

The output covariance matrix has dimensions p×p where element [i,j] is the covariance
between feature i and feature j.

Use cases:
- Feature analysis in machine learning (understanding feature relationships)
- Financial analysis (portfolio risk, asset correlations)
- Multivariate statistical analysis
- Principal Component Analysis (PCA) preprocessing

Time complexity: O(n*p²) - dominated by matrix multiplication X^T X
Space complexity: O(p²) - covariance matrix storage

Prerequisites:
- Input matrix must have at least 2 rows (samples)
- Input matrix must have at least 1 column (feature)
- Output matrix must be p×p where p is the number of columns in input
- allocFn must be provided for creating temporary matrices

Edge cases:
- Panics if matrix has < 2 rows
- Panics if output matrix dimensions don't match (must be p×p)
- Sample covariance uses Bessel's correction (divides by n-1 instead of n)

Algorithm:
1. Compute column means (one pass through matrix)
2. Mean-center matrix: X_centered = X - μ (broadcast column means)
3. Compute X^T X using matrix multiplication (leverages Blaze optimization)
4. Divide by (n-1) for sample or n for population

The implementation leverages Blaze's matrix multiplication for optimal performance,
providing massive speedups over nested loops of pairwise vector operations.
*/
func StatArchMultivariateCovarianceMatrixF32[T foundation.Numeric](
	matrixX memcore.MarkRaw,
	outputCovMatrix memcore.MarkRaw,
	sample bool,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) {
	rows := memstruct.MatrixRowsGet[T](matrixX)
	cols := memstruct.MatrixColsGet[T](matrixX)

	if rows < 2 {
		panic("cannot compute covariance matrix with less than 2 samples")
	}
	if cols == 0 {
		panic("cannot compute covariance matrix with 0 features")
	}

	// Validate output matrix dimensions (must be p×p)
	outputRows := memstruct.MatrixRowsGet[float32](outputCovMatrix)
	outputCols := memstruct.MatrixColsGet[float32](outputCovMatrix)
	if outputRows != cols || outputCols != cols {
		panic("output covariance matrix must be p×p where p is the number of columns in input matrix")
	}

	// Step 1: Compute column means
	columnMeans, _ := memarch.MemArchVectorCreate[float32](allocFn, cols)
	for col := uint64(0); col < cols; col++ {
		var sum float32 = 0
		for row := uint64(0); row < rows; row++ {
			val := memstruct.MatrixItemGetAtUnsafe[T](matrixX, row, col)
			sum += float32(val)
		}
		mean := sum / float32(rows)
		memstruct.VectorSetAtUnsafe(columnMeans, col, mean)
	}

	// Step 2: Mean-center the matrix
	// Create a copy of the matrix and subtract column means
	centeredMatrix, _ := memarch.MemArchMatrixCreate[float32](allocFn, rows, cols)
	for row := uint64(0); row < rows; row++ {
		for col := uint64(0); col < cols; col++ {
			val := float32(memstruct.MatrixItemGetAtUnsafe[T](matrixX, row, col))
			mean := memstruct.VectorItemGetAtUnsafe[float32](columnMeans, col)
			memstruct.MatrixSetAtUnsafe(centeredMatrix, row, col, val-mean)
		}
	}

	// Step 3: Compute X^T X
	// First transpose the centered matrix
	transposedMatrix, _ := memarch.MemArchMatrixCreate[float32](allocFn, cols, rows)
	structure.BlazeStructureMatrixTranspose[float32](centeredMatrix, transposedMatrix)

	// Then multiply: X^T × X
	// transposedMatrix is cols×rows, centeredMatrix is rows×cols
	// Result is cols×cols (which matches outputCovMatrix)
	structure.BlazeStructureMatrixMultiplyF32[float32, float32](transposedMatrix, centeredMatrix, outputCovMatrix)

	// Step 4: Divide by (n-1) for sample or n for population
	denominator := float32(rows)
	if sample {
		denominator = float32(rows) - 1
	}
	scalar.BlazeScalarMatrixDivideF32[float32](outputCovMatrix, outputCovMatrix, denominator)
}

/*
StatArchMultivariateCovarianceMatrixF64 computes the covariance matrix for a data matrix in float64 precision.

See StatArchMultivariateCovarianceMatrixF32 for detailed documentation.
This is the float64 variant for higher precision requirements.
*/
func StatArchMultivariateCovarianceMatrixF64[T foundation.Numeric](
	matrixX memcore.MarkRaw,
	outputCovMatrix memcore.MarkRaw,
	sample bool,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) {
	rows := memstruct.MatrixRowsGet[T](matrixX)
	cols := memstruct.MatrixColsGet[T](matrixX)

	if rows < 2 {
		panic("cannot compute covariance matrix with less than 2 samples")
	}
	if cols == 0 {
		panic("cannot compute covariance matrix with 0 features")
	}

	// Validate output matrix dimensions (must be p×p)
	outputRows := memstruct.MatrixRowsGet[float64](outputCovMatrix)
	outputCols := memstruct.MatrixColsGet[float64](outputCovMatrix)
	if outputRows != cols || outputCols != cols {
		panic("output covariance matrix must be p×p where p is the number of columns in input matrix")
	}

	// Step 1: Compute column means
	columnMeans, _ := memarch.MemArchVectorCreate[float64](allocFn, cols)
	for col := uint64(0); col < cols; col++ {
		var sum float64 = 0
		for row := uint64(0); row < rows; row++ {
			val := memstruct.MatrixItemGetAtUnsafe[T](matrixX, row, col)
			sum += float64(val)
		}
		mean := sum / float64(rows)
		memstruct.VectorSetAtUnsafe(columnMeans, col, mean)
	}

	// Step 2: Mean-center the matrix
	// Create a copy of the matrix and subtract column means
	centeredMatrix, _ := memarch.MemArchMatrixCreate[float64](allocFn, rows, cols)
	for row := uint64(0); row < rows; row++ {
		for col := uint64(0); col < cols; col++ {
			val := float64(memstruct.MatrixItemGetAtUnsafe[T](matrixX, row, col))
			mean := memstruct.VectorItemGetAtUnsafe[float64](columnMeans, col)
			memstruct.MatrixSetAtUnsafe(centeredMatrix, row, col, val-mean)
		}
	}

	// Step 3: Compute X^T X
	// First transpose the centered matrix
	transposedMatrix, _ := memarch.MemArchMatrixCreate[float64](allocFn, cols, rows)
	structure.BlazeStructureMatrixTranspose[float64](centeredMatrix, transposedMatrix)

	// Then multiply: X^T × X
	// transposedMatrix is cols×rows, centeredMatrix is rows×cols
	// Result is cols×cols (which matches outputCovMatrix)
	structure.BlazeStructureMatrixMultiplyF64[float64, float64](transposedMatrix, centeredMatrix, outputCovMatrix)

	// Step 4: Divide by (n-1) for sample or n for population
	denominator := float64(rows)
	if sample {
		denominator = float64(rows) - 1
	}
	scalar.BlazeScalarMatrixDivideF64[float64](outputCovMatrix, outputCovMatrix, denominator)
}

/*
StatArchMultivariateCorrelationMatrixF32 computes the correlation matrix for a data matrix in float32 precision.

The input matrix X has dimensions n×p where:
- n is the number of samples (rows)
- p is the number of features (columns)

The output correlation matrix has dimensions p×p where element [i,j] is the Pearson
correlation coefficient between feature i and feature j.

Use cases:
- Feature analysis in machine learning (understanding feature relationships)
- Financial analysis (asset correlations, portfolio diversification)
- Multivariate statistical analysis
- Principal Component Analysis (PCA) preprocessing

Time complexity: O(n*p²) - dominated by covariance matrix computation
Space complexity: O(p²) - correlation matrix storage

Prerequisites:
- Input matrix must have at least 2 rows (samples)
- Input matrix must have at least 1 column (feature)
- Output matrix must be p×p where p is the number of columns in input
- allocFn must be provided for creating temporary matrices

Edge cases:
- Panics if matrix has < 2 rows
- Panics if output matrix dimensions don't match (must be p×p)
- Panics if any column has zero variance (cannot compute correlation)
- Diagonal elements are always 1.0 (perfect correlation with itself)

Algorithm:
1. Compute covariance matrix (reuse StatArchMultivariateCovarianceMatrixF32)
2. Extract diagonal (variances) and compute standard deviations
3. Normalize: corr[i,j] = cov[i,j] / (σ[i] * σ[j])
4. Set diagonal to 1.0

The implementation leverages the covariance matrix computation for optimal performance.
*/
func StatArchMultivariateCorrelationMatrixF32[T foundation.Numeric](
	matrixX memcore.MarkRaw,
	outputCorrMatrix memcore.MarkRaw,
	sample bool,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) {
	rows := memstruct.MatrixRowsGet[T](matrixX)
	cols := memstruct.MatrixColsGet[T](matrixX)

	if rows < 2 {
		panic("cannot compute correlation matrix with less than 2 samples")
	}
	if cols == 0 {
		panic("cannot compute correlation matrix with 0 features")
	}

	// Validate output matrix dimensions (must be p×p)
	outputRows := memstruct.MatrixRowsGet[float32](outputCorrMatrix)
	outputCols := memstruct.MatrixColsGet[float32](outputCorrMatrix)
	if outputRows != cols || outputCols != cols {
		panic("output correlation matrix must be p×p where p is the number of columns in input matrix")
	}

	// Step 1: Compute covariance matrix
	covMatrix, _ := memarch.MemArchMatrixCreate[float32](allocFn, cols, cols)
	StatArchMultivariateCovarianceMatrixF32[T](matrixX, covMatrix, sample, allocFn)

	// Step 2: Extract diagonal (variances) and compute standard deviations
	stddevs, _ := memarch.MemArchVectorCreate[float32](allocFn, cols)
	for i := uint64(0); i < cols; i++ {
		variance := memstruct.MatrixItemGetAtUnsafe[float32](covMatrix, i, i)
		if variance < 0 {
			panic("covariance matrix has negative diagonal (numerical error)")
		}
		if variance == 0 {
			panic("cannot compute correlation: column has zero variance")
		}
		stddev := foundation.Sqrt32(variance)
		memstruct.VectorSetAtUnsafe(stddevs, i, stddev)
	}

	// Step 3: Normalize covariance matrix to get correlation matrix
	// corr[i,j] = cov[i,j] / (σ[i] * σ[j])
	for i := uint64(0); i < cols; i++ {
		stddevI := memstruct.VectorItemGetAtUnsafe[float32](stddevs, i)
		for j := uint64(0); j < cols; j++ {
			stddevJ := memstruct.VectorItemGetAtUnsafe[float32](stddevs, j)
			cov := memstruct.MatrixItemGetAtUnsafe[float32](covMatrix, i, j)
			corr := cov / (stddevI * stddevJ)
			memstruct.MatrixSetAtUnsafe(outputCorrMatrix, i, j, corr)
		}
	}

	// Step 4: Set diagonal to 1.0 (perfect correlation with itself)
	for i := uint64(0); i < cols; i++ {
		memstruct.MatrixSetAtUnsafe(outputCorrMatrix, i, i, 1.0)
	}
}

/*
StatArchMultivariateCorrelationMatrixF64 computes the correlation matrix for a data matrix in float64 precision.

See StatArchMultivariateCorrelationMatrixF32 for detailed documentation.
This is the float64 variant for higher precision requirements.
*/
func StatArchMultivariateCorrelationMatrixF64[T foundation.Numeric](
	matrixX memcore.MarkRaw,
	outputCorrMatrix memcore.MarkRaw,
	sample bool,
	allocFn func(sizeBytes, alignment uint64) memcore.MarkRaw,
) {
	rows := memstruct.MatrixRowsGet[T](matrixX)
	cols := memstruct.MatrixColsGet[T](matrixX)

	if rows < 2 {
		panic("cannot compute correlation matrix with less than 2 samples")
	}
	if cols == 0 {
		panic("cannot compute correlation matrix with 0 features")
	}

	// Validate output matrix dimensions (must be p×p)
	outputRows := memstruct.MatrixRowsGet[float64](outputCorrMatrix)
	outputCols := memstruct.MatrixColsGet[float64](outputCorrMatrix)
	if outputRows != cols || outputCols != cols {
		panic("output correlation matrix must be p×p where p is the number of columns in input matrix")
	}

	// Step 1: Compute covariance matrix
	covMatrix, _ := memarch.MemArchMatrixCreate[float64](allocFn, cols, cols)
	StatArchMultivariateCovarianceMatrixF64[T](matrixX, covMatrix, sample, allocFn)

	// Step 2: Extract diagonal (variances) and compute standard deviations
	stddevs, _ := memarch.MemArchVectorCreate[float64](allocFn, cols)
	for i := uint64(0); i < cols; i++ {
		variance := memstruct.MatrixItemGetAtUnsafe[float64](covMatrix, i, i)
		if variance < 0 {
			panic("covariance matrix has negative diagonal (numerical error)")
		}
		if variance == 0 {
			panic("cannot compute correlation: column has zero variance")
		}
		stddev := foundation.Sqrt64(variance)
		memstruct.VectorSetAtUnsafe(stddevs, i, stddev)
	}

	// Step 3: Normalize covariance matrix to get correlation matrix
	// corr[i,j] = cov[i,j] / (σ[i] * σ[j])
	for i := uint64(0); i < cols; i++ {
		stddevI := memstruct.VectorItemGetAtUnsafe[float64](stddevs, i)
		for j := uint64(0); j < cols; j++ {
			stddevJ := memstruct.VectorItemGetAtUnsafe[float64](stddevs, j)
			cov := memstruct.MatrixItemGetAtUnsafe[float64](covMatrix, i, j)
			corr := cov / (stddevI * stddevJ)
			memstruct.MatrixSetAtUnsafe(outputCorrMatrix, i, j, corr)
		}
	}

	// Step 4: Set diagonal to 1.0 (perfect correlation with itself)
	for i := uint64(0); i < cols; i++ {
		memstruct.MatrixSetAtUnsafe(outputCorrMatrix, i, i, 1.0)
	}
}



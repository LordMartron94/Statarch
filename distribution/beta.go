package distribution

import (
	blazemath "blaze/math"
	"math"
)

const (
	// betaQuantileMaxIterations limits Newton-Raphson iterations for quantile computation
	betaQuantileMaxIterations = 50

	// betaQuantileTolerance is the convergence tolerance for quantile computation
	betaQuantileTolerance = 1.0e-10

	// betaQuantileBinarySearchFallbackThreshold is the minimum improvement required
	// to continue Newton-Raphson, otherwise fall back to binary search
	betaQuantileBinarySearchFallbackThreshold = 1.0e-6
)

/*
StatArchDistributionBetaCDFF32 computes the cumulative distribution function (CDF) of the Beta distribution in float32 precision.

The Beta distribution CDF gives the probability that a Beta-distributed random variable
is less than or equal to x: P(X ≤ x).

Use cases:
- Statistical hypothesis testing
- Bayesian inference
- Confidence interval construction
- Quantile estimation (used internally by Quantile function)

Time complexity: O(1) - direct call to optimized beta function
Space complexity: O(1) - only local variables used

Prerequisites:
- x must be in range [0, 1]
- alpha > 0
- beta > 0

Edge cases:
- Panics if x < 0 or x > 1
- Panics if alpha <= 0 or beta <= 0
- Returns 0 if x = 0
- Returns 1 if x = 1

Formula: CDF(x) = I_x(α, β) where I_x is the regularized incomplete beta function.

The implementation uses Blaze's optimized beta function which implements the Lentz
continued fraction method for high accuracy even in the tails.
*/
func StatArchDistributionBetaCDFF32(x, alpha, beta float32) float32 {
	if x < 0 || x > 1 {
		panic("StatArchDistributionBetaCDFF32: x must be in [0, 1]")
	}
	if alpha <= 0 || beta <= 0 {
		panic("StatArchDistributionBetaCDFF32: alpha and beta must be > 0")
	}

	if x == 0 {
		return 0.0
	}
	if x == 1 {
		return 1.0
	}

	return blazemath.BlazeMathBetaRegularizedIncompleteF32(x, alpha, beta)
}

/*
StatArchDistributionBetaCDFF64 computes the cumulative distribution function (CDF) of the Beta distribution in float64 precision.

See StatArchDistributionBetaCDFF32 for detailed documentation.
This is the float64 variant for higher precision requirements.
*/
func StatArchDistributionBetaCDFF64(x, alpha, beta float64) float64 {
	if x < 0 || x > 1 {
		panic("StatArchDistributionBetaCDFF64: x must be in [0, 1]")
	}
	if alpha <= 0 || beta <= 0 {
		panic("StatArchDistributionBetaCDFF64: alpha and beta must be > 0")
	}

	if x == 0 {
		return 0.0
	}
	if x == 1 {
		return 1.0
	}

	return blazemath.BlazeMathBetaRegularizedIncompleteF64(x, alpha, beta)
}

/*
StatArchDistributionBetaPDFF32 computes the probability density function (PDF) of the Beta distribution in float32 precision.

The Beta distribution PDF gives the probability density at point x.

Use cases:
- Statistical modeling
- Bayesian inference
- Maximum likelihood estimation
- Distribution visualization

Time complexity: O(1) - uses log-space computation for numerical stability
Space complexity: O(1) - only local variables used

Prerequisites:
- x must be in range (0, 1) for standard cases
- alpha > 0
- beta > 0

Edge cases:
- Panics if x < 0 or x > 1
- Panics if alpha <= 0 or beta <= 0
- For x = 0: returns 0 if alpha > 1, +Inf if alpha < 1, beta if alpha = 1
- For x = 1: returns 0 if beta > 1, +Inf if beta < 1, alpha if beta = 1

Formula: PDF(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)
where B(α,β) = Γ(α) * Γ(β) / Γ(α+β) is the Beta function.

The implementation uses log-space computation for numerical stability:
ln(PDF) = (α-1)ln(x) + (β-1)ln(1-x) - ln(B(α,β))
*/
func StatArchDistributionBetaPDFF32(x, alpha, beta float32) float32 {
	if x < 0 || x > 1 {
		panic("StatArchDistributionBetaPDFF32: x must be in [0, 1]")
	}
	if alpha <= 0 || beta <= 0 {
		panic("StatArchDistributionBetaPDFF32: alpha and beta must be > 0")
	}

	// Handle boundary cases
	if x == 0 {
		if alpha > 1 {
			return 0.0
		}
		if alpha < 1 {
			return float32(math.Inf(1))
		}
		// alpha == 1: PDF(0) = (1-x)^(β-1) / B(1,β) = 1 / B(1,β) = β
		return beta
	}
	if x == 1 {
		if beta > 1 {
			return 0.0
		}
		if beta < 1 {
			return float32(math.Inf(1))
		}
		// beta == 1: PDF(1) = x^(α-1) / B(α,1) = 1 / B(α,1) = α
		return alpha
	}

	// Compute in log-space for numerical stability
	// ln(PDF) = (α-1)ln(x) + (β-1)ln(1-x) - ln(B(α,β))
	// where ln(B(α,β)) = ln(Γ(α)) + ln(Γ(β)) - ln(Γ(α+β))
	alpha64 := float64(alpha)
	beta64 := float64(beta)
	x64 := float64(x)

	lgammaAlpha := blazemath.BlazeMathLogGammaAbsF64(alpha64)
	lgammaBeta := blazemath.BlazeMathLogGammaAbsF64(beta64)
	lgammaAlphaBeta := blazemath.BlazeMathLogGammaAbsF64(alpha64 + beta64)

	lnBeta := lgammaAlpha + lgammaBeta - lgammaAlphaBeta

	lnPDF := (alpha64-1)*math.Log(x64) + (beta64-1)*math.Log(1.0-x64) - lnBeta

	return float32(math.Exp(lnPDF))
}

/*
StatArchDistributionBetaPDFF64 computes the probability density function (PDF) of the Beta distribution in float64 precision.

See StatArchDistributionBetaPDFF32 for detailed documentation.
This is the float64 variant for higher precision requirements.
*/
func StatArchDistributionBetaPDFF64(x, alpha, beta float64) float64 {
	if x < 0 || x > 1 {
		panic("StatArchDistributionBetaPDFF64: x must be in [0, 1]")
	}
	if alpha <= 0 || beta <= 0 {
		panic("StatArchDistributionBetaPDFF64: alpha and beta must be > 0")
	}

	// Handle boundary cases
	if x == 0 {
		if alpha > 1 {
			return 0.0
		}
		if alpha < 1 {
			return math.Inf(1)
		}
		// alpha == 1: PDF(0) = (1-x)^(β-1) / B(1,β) = 1 / B(1,β) = β
		return beta
	}
	if x == 1 {
		if beta > 1 {
			return 0.0
		}
		if beta < 1 {
			return math.Inf(1)
		}
		// beta == 1: PDF(1) = x^(α-1) / B(α,1) = 1 / B(α,1) = α
		return alpha
	}

	// Compute in log-space for numerical stability
	// ln(PDF) = (α-1)ln(x) + (β-1)ln(1-x) - ln(B(α,β))
	// where ln(B(α,β)) = ln(Γ(α)) + ln(Γ(β)) - ln(Γ(α+β))
	lgammaAlpha := blazemath.BlazeMathLogGammaAbsF64(alpha)
	lgammaBeta := blazemath.BlazeMathLogGammaAbsF64(beta)
	lgammaAlphaBeta := blazemath.BlazeMathLogGammaAbsF64(alpha + beta)

	lnBeta := lgammaAlpha + lgammaBeta - lgammaAlphaBeta

	lnPDF := (alpha-1)*math.Log(x) + (beta-1)*math.Log(1.0-x) - lnBeta

	return math.Exp(lnPDF)
}

/*
StatArchDistributionBetaQuantileF32 computes the quantile (inverse CDF) of the Beta distribution in float32 precision.

The quantile function returns the value x such that P(X ≤ x) = p, where p is the probability.

Use cases:
- Confidence interval construction
- Hypothesis testing
- Random variate generation (inverse transform sampling)
- Statistical modeling

Time complexity: O(k) where k is the number of Newton-Raphson iterations (typically 3-5)
Space complexity: O(1) - only local variables used

Prerequisites:
- p must be in range [0, 1]
- alpha > 0
- beta > 0

Edge cases:
- Panics if p < 0 or p > 1
- Panics if alpha <= 0 or beta <= 0
- Returns 0 if p = 0
- Returns 1 if p = 1

Algorithm:
Uses Newton-Raphson iteration to solve CDF(x) = p:
- Start with initial guess: x = α/(α+β) (mean of Beta distribution)
- Iterate: x_new = x_old - (CDF(x_old) - p) / PDF(x_old)
- Converge when |CDF(x) - p| < tolerance
- Fallback to binary search if Newton-Raphson fails to converge or makes insufficient progress

The implementation uses Blaze's optimized beta function for CDF computation
and log-space PDF computation for numerical stability.
*/
func StatArchDistributionBetaQuantileF32(p, alpha, beta float32) float32 {
	if p < 0 || p > 1 {
		panic("StatArchDistributionBetaQuantileF32: p must be in [0, 1]")
	}
	if alpha <= 0 || beta <= 0 {
		panic("StatArchDistributionBetaQuantileF32: alpha and beta must be > 0")
	}

	// Handle boundary cases
	if p == 0 {
		return 0.0
	}
	if p == 1 {
		return 1.0
	}

	alpha64 := float64(alpha)
	beta64 := float64(beta)
	p64 := float64(p)

	// Initial guess: mean of Beta distribution
	x := alpha64 / (alpha64 + beta64)

	// Clamp initial guess to valid range
	if x <= 0 {
		x = 0.01
	}
	if x >= 1 {
		x = 0.99
	}

	// Newton-Raphson iteration
	for iter := 0; iter < betaQuantileMaxIterations; iter++ {
		cdf := blazemath.BlazeMathBetaRegularizedIncompleteF64(x, alpha64, beta64)
		error := cdf - p64

		// Check convergence
		if math.Abs(error) < betaQuantileTolerance {
			return float32(x)
		}

		// Compute PDF for Newton-Raphson step
		pdf := StatArchDistributionBetaPDFF64(x, alpha64, beta64)

		// Avoid division by zero
		if pdf == 0 {
			// Fall back to binary search
			return float32(betaQuantileBinarySearchF64(p64, alpha64, beta64))
		}

		// Newton-Raphson step: x_new = x_old - (CDF(x) - p) / PDF(x)
		xNew := x - error/pdf

		// Check for insufficient progress (fallback to binary search)
		if math.Abs(xNew-x) < betaQuantileBinarySearchFallbackThreshold && math.Abs(error) > betaQuantileTolerance {
			return float32(betaQuantileBinarySearchF64(p64, alpha64, beta64))
		}

		// Clamp to valid range
		if xNew <= 0 {
			xNew = 1.0e-10
		}
		if xNew >= 1 {
			xNew = 1.0 - 1.0e-10
		}

		x = xNew
	}

	// If Newton-Raphson didn't converge, fall back to binary search
	return float32(betaQuantileBinarySearchF64(p64, alpha64, beta64))
}

/*
StatArchDistributionBetaQuantileF64 computes the quantile (inverse CDF) of the Beta distribution in float64 precision.

See StatArchDistributionBetaQuantileF32 for detailed documentation.
This is the float64 variant for higher precision requirements.
*/
func StatArchDistributionBetaQuantileF64(p, alpha, beta float64) float64 {
	if p < 0 || p > 1 {
		panic("StatArchDistributionBetaQuantileF64: p must be in [0, 1]")
	}
	if alpha <= 0 || beta <= 0 {
		panic("StatArchDistributionBetaQuantileF64: alpha and beta must be > 0")
	}

	// Handle boundary cases
	if p == 0 {
		return 0.0
	}
	if p == 1 {
		return 1.0
	}

	// Initial guess: mean of Beta distribution
	x := alpha / (alpha + beta)

	// Clamp initial guess to valid range
	if x <= 0 {
		x = 0.01
	}
	if x >= 1 {
		x = 0.99
	}

	// Newton-Raphson iteration
	for iter := 0; iter < betaQuantileMaxIterations; iter++ {
		cdf := blazemath.BlazeMathBetaRegularizedIncompleteF64(x, alpha, beta)
		error := cdf - p

		// Check convergence
		if math.Abs(error) < betaQuantileTolerance {
			return x
		}

		// Compute PDF for Newton-Raphson step
		pdf := StatArchDistributionBetaPDFF64(x, alpha, beta)

		// Avoid division by zero
		if pdf == 0 {
			// Fall back to binary search
			return betaQuantileBinarySearchF64(p, alpha, beta)
		}

		// Newton-Raphson step: x_new = x_old - (CDF(x) - p) / PDF(x)
		xNew := x - error/pdf

		// Check for insufficient progress (fallback to binary search)
		if math.Abs(xNew-x) < betaQuantileBinarySearchFallbackThreshold && math.Abs(error) > betaQuantileTolerance {
			return betaQuantileBinarySearchF64(p, alpha, beta)
		}

		// Clamp to valid range
		if xNew <= 0 {
			xNew = 1.0e-10
		}
		if xNew >= 1 {
			xNew = 1.0 - 1.0e-10
		}

		x = xNew
	}

	// If Newton-Raphson didn't converge, fall back to binary search
	return betaQuantileBinarySearchF64(p, alpha, beta)
}

// betaQuantileBinarySearchF64 performs binary search to find quantile when Newton-Raphson fails.
func betaQuantileBinarySearchF64(p, alpha, beta float64) float64 {
	low := 0.0
	high := 1.0
	mid := 0.5

	for iter := 0; iter < betaQuantileMaxIterations; iter++ {
		cdf := blazemath.BlazeMathBetaRegularizedIncompleteF64(mid, alpha, beta)
		error := cdf - p

		if math.Abs(error) < betaQuantileTolerance {
			return mid
		}

		if error > 0 {
			// CDF is too high, need smaller x
			high = mid
		} else {
			// CDF is too low, need larger x
			low = mid
		}

		mid = (low + high) / 2.0

		// Check if we've converged to boundaries
		if high-low < betaQuantileTolerance {
			return mid
		}
	}

	return mid
}



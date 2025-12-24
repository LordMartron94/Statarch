package rolling

import (
	"foundation"
)

/*
StatArchRollingExponentialMovingAverage holds state for an exponentially weighted moving average.

EWMA provides an "infinite" window with zero memory growth, making it ideal for long-term trend
analysis in resource-constrained environments. Unlike fixed-window rolling statistics, EWMA
does not require storing historical values.

Time complexity: O(1) per update
Space complexity: O(1) - constant memory regardless of data stream length

Use cases:
- Long-term trend analysis without fixed window size
- Noise reduction in streaming data
- Resource-constrained environments where circular buffers are too expensive
- Dampening noise in real-time monitoring

The structure maintains a single value that is updated incrementally using exponential smoothing.
*/
type StatArchRollingExponentialMovingAverage[T foundation.Numeric] struct {
	Value       float64 // Current EWMA value
	Alpha       float64 // Smoothing factor (0 < alpha <= 1)
	Initialized bool    // Whether value has been initialized
}

/*
StatArchRollingExponentialMovingAverageCreate creates a new EWMA structure.

Time complexity: O(1)
Space complexity: O(1)

Parameters:
- alpha: Smoothing factor in range (0, 1]. Smaller values provide more smoothing (slower response to changes).
  Typical values: 0.1-0.3 for heavy smoothing, 0.5-0.9 for lighter smoothing.

Returns:
- A new EWMA structure ready for use

The structure is initialized with:
- Value = 0
- Alpha = provided alpha (validated)
- Initialized = false

Edge cases:
- Panics if alpha <= 0 or alpha > 1
*/
func StatArchRollingExponentialMovingAverageCreate[T foundation.Numeric](
	alpha float64,
) *StatArchRollingExponentialMovingAverage[T] {
	if alpha <= 0 || alpha > 1 {
		panic("StatArchRollingExponentialMovingAverageCreate: alpha must be in (0, 1]")
	}

	return &StatArchRollingExponentialMovingAverage[T]{
		Value:       0,
		Alpha:       alpha,
		Initialized: false,
	}
}

/*
StatArchRollingExponentialMovingAverageUpdate updates the EWMA with a new value.

The update formula is: EWMA = alpha * value + (1 - alpha) * EWMA_prev

On the first update, the EWMA is initialized to the value (no smoothing applied).
Subsequent updates use exponential smoothing.

Time complexity: O(1)
Space complexity: O(1)

Parameters:
- ewma: The EWMA structure to update
- value: The new value to incorporate into the average

The function modifies the EWMA structure in place.
*/
func StatArchRollingExponentialMovingAverageUpdate[T foundation.Numeric](
	ewma *StatArchRollingExponentialMovingAverage[T],
	value T,
) {
	val := float64(value)

	if !ewma.Initialized {
		// First update: initialize to the value
		ewma.Value = val
		ewma.Initialized = true
		return
	}

	// Exponential smoothing: EWMA = alpha * value + (1 - alpha) * EWMA_prev
	ewma.Value = ewma.Alpha*val + (1-ewma.Alpha)*ewma.Value
}

/*
StatArchRollingExponentialMovingAverageGetF32 returns the current EWMA value in float32 precision.

Time complexity: O(1)
Space complexity: O(1)

Returns:
- The current EWMA value as float32
- Returns 0 if no values have been added yet (not initialized)
*/
func StatArchRollingExponentialMovingAverageGetF32[T foundation.Numeric](
	ewma *StatArchRollingExponentialMovingAverage[T],
) float32 {
	if !ewma.Initialized {
		return 0
	}
	return float32(ewma.Value)
}

/*
StatArchRollingExponentialMovingAverageGetF64 returns the current EWMA value in float64 precision.

Time complexity: O(1)
Space complexity: O(1)

Returns:
- The current EWMA value as float64
- Returns 0 if no values have been added yet (not initialized)
*/
func StatArchRollingExponentialMovingAverageGetF64[T foundation.Numeric](
	ewma *StatArchRollingExponentialMovingAverage[T],
) float64 {
	if !ewma.Initialized {
		return 0
	}
	return ewma.Value
}


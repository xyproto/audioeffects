// audioeffects.go
// Package audioeffects provides a collection of audio processing effects for manipulating audio samples.
package audioeffects

import (
	"math"
)

// Biquad represents a biquad filter with its coefficients and state variables.
type Biquad struct {
	a0, a1, a2 float64 // Feedforward coefficients
	b1, b2     float64 // Feedback coefficients
	x1, x2     float64 // Previous input samples
	y1, y2     float64 // Previous output samples
}

// NewBiquad creates a new Biquad filter based on the specified type and parameters.
// filterType can be "low-pass", "high-pass", "band-pass", "notch", etc.
// freq is the center frequency, Q is the quality factor, and sampleRate is the sampling rate in Hz.
func NewBiquad(filterType string, freq, Q, sampleRate float64) *Biquad {
	omega := 2 * math.Pi * freq / sampleRate
	sinOmega := math.Sin(omega)
	cosOmega := math.Cos(omega)
	alpha := sinOmega / (2 * Q)

	var a0, a1, a2, b1, b2 float64

	switch filterType {
	case "low-pass":
		a0 = 1 + alpha
		a1 = -2 * cosOmega
		a2 = 1 - alpha
		b1 = 2 * cosOmega
		b2 = 1 - alpha
	case "high-pass":
		a0 = 1 + alpha
		a1 = -2 * cosOmega
		a2 = 1 - alpha
		b1 = 2 * cosOmega
		b2 = 1 - alpha
	case "band-pass":
		a0 = 1 + alpha
		a1 = -2 * cosOmega
		a2 = 1 - alpha
		b1 = 2 * cosOmega
		b2 = 1 - alpha
	case "notch":
		a0 = 1 + alpha
		a1 = -2 * cosOmega
		a2 = 1 - alpha
		b1 = 2 * cosOmega
		b2 = 1 - alpha
	// Add other filter types as needed
	default:
		// Default to pass-through if filter type is unknown
		a0 = 1
		a1 = 0
		a2 = 0
		b1 = 0
		b2 = 0
	}

	// Normalize coefficients
	return &Biquad{
		a0: a0 / a0, // This will always be 1.0; kept for consistency
		a1: a1 / a0,
		a2: a2 / a0,
		b1: b1 / a0,
		b2: b2 / a0,
		x1: 0,
		x2: 0,
		y1: 0,
		y2: 0,
	}
}

// UpdateParameters updates the filter coefficients based on new parameters.
// This allows dynamic modification of the filter in real-time.
func (b *Biquad) UpdateParameters(filterType string, freq, Q, sampleRate float64) {
	omega := 2 * math.Pi * freq / sampleRate
	sinOmega := math.Sin(omega)
	cosOmega := math.Cos(omega)
	alpha := sinOmega / (2 * Q)

	var a0, a1, a2, b1, b2 float64

	switch filterType {
	case "low-pass":
		a0 = 1 + alpha
		a1 = -2 * cosOmega
		a2 = 1 - alpha
		b1 = 2 * cosOmega
		b2 = 1 - alpha
	case "high-pass":
		a0 = 1 + alpha
		a1 = -2 * cosOmega
		a2 = 1 - alpha
		b1 = 2 * cosOmega
		b2 = 1 - alpha
	case "band-pass":
		a0 = 1 + alpha
		a1 = -2 * cosOmega
		a2 = 1 - alpha
		b1 = 2 * cosOmega
		b2 = 1 - alpha
	case "notch":
		a0 = 1 + alpha
		a1 = -2 * cosOmega
		a2 = 1 - alpha
		b1 = 2 * cosOmega
		b2 = 1 - alpha
	// Add other filter types as needed
	default:
		// Default to pass-through if filter type is unknown
		a0 = 1
		a1 = 0
		a2 = 0
		b1 = 0
		b2 = 0
	}

	// Normalize coefficients
	b.a0 = a0 / a0 // This will always be 1.0; kept for consistency
	b.a1 = a1 / a0
	b.a2 = a2 / a0
	b.b1 = b1 / a0
	b.b2 = b2 / a0
}

// Process applies the biquad filter to a single sample.
func (b *Biquad) Process(sample float64) float64 {
	y := b.a0*sample + b.a1*b.x1 + b.a2*b.x2 - b.b1*b.y1 - b.b2*b.y2
	b.x2 = b.x1
	b.x1 = sample
	b.y2 = b.y1
	b.y1 = y
	return y
}

// CompressorSettings represents a dynamic range compressor with its parameters.
type CompressorSettings struct {
	Threshold float64 // Threshold level above which compression is applied
	Ratio     float64 // Compression ratio
	Attack    float64 // Attack time in seconds
	Release   float64 // Release time in seconds
}

// Envelope applies an ADSR envelope to the waveform.
func Envelope(samples []float64, attack, decay, sustain, release float64, sampleRate int) []float64 {
	adsr := make([]float64, len(samples))
	totalDuration := float64(len(samples)) / float64(sampleRate)
	for i := range samples {
		t := float64(i) / float64(sampleRate)
		var envelope float64
		if t < attack {
			envelope = t / attack
		} else if t < attack+decay {
			envelope = 1 - (t-attack)/decay*(1-sustain)
		} else if t < totalDuration-release {
			envelope = sustain
		} else if t < totalDuration {
			envelope = sustain * (1 - (t-(totalDuration-release))/release)
		} else {
			envelope = 0.0
		}
		adsr[i] = samples[i] * envelope
	}
	return adsr
}

// EnvelopeAtTime generates the ADSR envelope value at a specific time.
func EnvelopeAtTime(t, attack, decay, sustain, release, duration float64) float64 {
	if t < attack {
		return t / attack
	}
	if t < attack+decay {
		return 1.0 - (t-attack)/decay*(1.0-sustain)
	}
	if t < duration-release {
		return sustain
	}
	if t < duration {
		return sustain * (1.0 - (t-(duration-release))/release)
	}
	return 0.0
}

// Drive applies a drive (distortion) effect to a single sample.
func Drive(sample, drive float64) float64 {
	if drive > 0 {
		return sample * (1 + drive) / (1 + drive*math.Abs(sample))
	}
	return sample
}

// PitchModulation applies pitch modulation to the samples using phase modulation.
func PitchModulation(samples []float64, modFreq, modDepth float64, sampleRate int) []float64 {
	modulated := make([]float64, len(samples))
	carrierPhase := 0.0
	modulatorPhase := 0.0
	for i := range samples {
		carrierFreq := modDepth * math.Sin(2*math.Pi*modFreq*modulatorPhase)
		carrierPhase += carrierFreq / float64(sampleRate)
		modulated[i] = math.Sin(2*math.Pi*carrierPhase) * samples[i]
		// Clamp the value to prevent clipping
		if modulated[i] > 1.0 {
			modulated[i] = 1.0
		} else if modulated[i] < -1.0 {
			modulated[i] = -1.0
		}
		modulatorPhase += 1.0 / float64(sampleRate)
	}
	return modulated
}

// Panning applies stereo panning to the samples.
// pan should be in the range [-1, 1], where -1 is full left and 1 is full right.
func Panning(samples []float64, pan float64) ([]float64, []float64) {
	leftChannel := make([]float64, len(samples))
	rightChannel := make([]float64, len(samples))
	// Clamp pan to [-1, 1]
	if pan < -1.0 {
		pan = -1.0
	} else if pan > 1.0 {
		pan = 1.0
	}
	leftGain := math.Cos((pan + 1) * math.Pi / 4)  // Equal-power panning
	rightGain := math.Sin((pan + 1) * math.Pi / 4) // Equal-power panning

	for i := range samples {
		leftChannel[i] = samples[i] * leftGain
		rightChannel[i] = samples[i] * rightGain
	}

	return leftChannel, rightChannel
}

// FrequencyModulation applies frequency modulation to a waveform using a modulator frequency and depth.
func FrequencyModulation(samples []float64, modFreq, modDepth float64, sampleRate int) []float64 {
	modulated := make([]float64, len(samples))
	carrierPhase := 0.0
	modulatorPhase := 0.0
	for i := range samples {
		carrierFreq := modDepth * math.Sin(2*math.Pi*modFreq*modulatorPhase)
		carrierPhase += carrierFreq / float64(sampleRate)
		modulated[i] = math.Sin(2*math.Pi*carrierPhase) * samples[i]
		// Clamp the value to prevent clipping
		if modulated[i] > 1.0 {
			modulated[i] = 1.0
		} else if modulated[i] < -1.0 {
			modulated[i] = -1.0
		}
		modulatorPhase += 1.0 / float64(sampleRate)
	}
	return modulated
}

// FadeIn applies a linear fade-in to the start of the samples.
func FadeIn(samples []float64, fadeDuration float64, sampleRate int) []float64 {
	fadeSamples := int(fadeDuration * float64(sampleRate))
	if fadeSamples > len(samples) {
		fadeSamples = len(samples)
	}
	for i := 0; i < fadeSamples; i++ {
		if fadeSamples > 1 {
			t := float64(i) / float64(fadeSamples)
			multiplier := t
			samples[i] *= multiplier
		} else {
			// For fadeSamples=1, set the first sample to full amplitude
			samples[0] *= 1.0
		}
	}
	return samples
}

// FadeOut applies a linear fade-out to the end of the samples.
func FadeOut(samples []float64, fadeDuration float64, sampleRate int) []float64 {
	totalSamples := len(samples)
	fadeSamples := int(fadeDuration * float64(sampleRate))
	if fadeSamples > totalSamples {
		fadeSamples = totalSamples
	}
	for i := 0; i < fadeSamples; i++ {
		if fadeSamples > 1 {
			t := float64(i) / float64(fadeSamples-1) // Ensure t reaches 1.0 at the last sample
			multiplier := 1.0 - t                    // Linear fade
			index := totalSamples - fadeSamples + i
			if index >= 0 && index < totalSamples {
				samples[index] *= multiplier
			}
		} else {
			// For fadeSamples=1, set the last sample to 0
			index := totalSamples - 1
			if index >= 0 && index < totalSamples {
				samples[index] *= 0.0
			}
		}
	}
	return samples
}

// Chorus applies a chorus effect to the samples.
// delaySec: Base delay time in seconds.
// modDepth: Modulation depth as a fraction (e.g., 0.05 for 5%).
// rate: Modulation rate in Hz.
// mix: Mixing proportion of the delayed signal (0.0 to 1.0).
func Chorus(samples []float64, sampleRate int, delaySec float64, modDepth float64, rate float64, mix float64) []float64 {
	delaySamples := int(delaySec * float64(sampleRate))
	if delaySamples <= 0 {
		delaySamples = 1
	}
	modulated := make([]float64, len(samples))
	buffer := make([]float64, delaySamples)
	bufferIndex := 0

	phaseIncrement := rate / float64(sampleRate)
	phase := 0.0

	for i := 0; i < len(samples); i++ {
		// Modulate the delay time with a sine wave
		modulation := modDepth * math.Sin(2*math.Pi*phase)
		currentDelay := int(float64(delaySamples) * (1 + modulation))
		if currentDelay < 0 {
			currentDelay = 0
		} else if currentDelay >= delaySamples {
			currentDelay = delaySamples - 1
		}

		// Calculate buffer read index
		readIndex := (bufferIndex - currentDelay + delaySamples) % delaySamples
		delayedSample := buffer[readIndex]

		// Mix the original and delayed samples
		modulated[i] = samples[i]*(1-mix) + delayedSample*mix

		// Update buffer with current sample
		buffer[bufferIndex] = samples[i]

		// Increment buffer index
		bufferIndex = (bufferIndex + 1) % delaySamples

		// Increment phase
		phase += phaseIncrement
		if phase >= 1.0 {
			phase -= 1.0
		}
	}

	// Clamp to avoid clipping
	for i := range modulated {
		if modulated[i] > 1.0 {
			modulated[i] = 1.0
		} else if modulated[i] < -1.0 {
			modulated[i] = -1.0
		}
	}

	return modulated
}

// LowPassFilter applies a basic low-pass filter to the samples.
func LowPassFilter(samples []float64, cutoff float64, sampleRate int) []float64 {
	filtered := make([]float64, len(samples))
	rc := 1.0 / (2.0 * math.Pi * cutoff)
	dt := 1.0 / float64(sampleRate)
	alpha := dt / (rc + dt)

	if len(samples) == 0 {
		return samples
	}

	prev := samples[0]       // Initialize with the first sample
	filtered[0] = samples[0] // The first sample remains the same

	for i := 1; i < len(samples); i++ {
		filtered[i] = alpha*samples[i] + (1-alpha)*prev
		prev = filtered[i]
	}
	return filtered
}

// HighPassFilter applies a basic high-pass filter to the samples.
func HighPassFilter(samples []float64, cutoff float64, sampleRate int) []float64 {
	filtered := make([]float64, len(samples))
	rc := 1.0 / (2.0 * math.Pi * cutoff)
	dt := 1.0 / float64(sampleRate)
	alpha := rc / (rc + dt)

	if len(samples) == 0 {
		return samples
	}

	filtered[0] = samples[0]

	for i := 1; i < len(samples); i++ {
		filtered[i] = alpha * (filtered[i-1] + samples[i] - samples[i-1])
	}
	return filtered
}

// BandPassFilter applies a band-pass filter to the samples using low and high cutoff frequencies.
func BandPassFilter(samples []float64, lowCutoff, highCutoff float64, sampleRate int) []float64 {
	lowPassed := LowPassFilter(samples, highCutoff, sampleRate)
	return HighPassFilter(lowPassed, lowCutoff, sampleRate)
}

// Limiter ensures the signal doesn't exceed [-1, 1] range.
func Limiter(samples []float64) []float64 {
	limited := make([]float64, len(samples))
	for i, sample := range samples {
		if sample > 1.0 {
			limited[i] = 1.0
		} else if sample < -1.0 {
			limited[i] = -1.0
		} else {
			limited[i] = sample
		}
	}
	return limited
}

// NormalizeSamples scales the samples so the peak amplitude matches the given target peak.
func NormalizeSamples(samples []float64, targetPeak float64) []float64 {
	if targetPeak <= 0 {
		// Invalid target peak, return samples unmodified
		return samples
	}

	currentPeak := 0.0
	for _, sample := range samples {
		abs := math.Abs(sample)
		if abs > currentPeak {
			currentPeak = abs
		}
	}

	if currentPeak == 0 {
		return samples
	}

	scale := targetPeak / currentPeak
	normalizedSamples := make([]float64, len(samples))
	for i, sample := range samples {
		normalized := sample * scale
		// Clamp the values to the [-1, 1] range after scaling
		if normalized > 1.0 {
			normalizedSamples[i] = 1.0
		} else if normalized < -1.0 {
			normalizedSamples[i] = -1.0
		} else {
			normalizedSamples[i] = normalized
		}
	}
	return normalizedSamples
}

// Reverb applies a simple feedback reverb effect to the samples.
// delayTime: Delay time in seconds.
// decay: Decay factor for the reverb.
func Reverb(samples []float64, sampleRate int, delayTime float64, decay float64) []float64 {
	delaySamples := int(delayTime * float64(sampleRate))
	if delaySamples <= 0 {
		delaySamples = 1
	}
	reverb := make([]float64, len(samples))
	buffer := make([]float64, delaySamples)
	bufferIndex := 0

	for i, sample := range samples {
		delayed := buffer[bufferIndex]
		reverb[i] = sample + delayed*decay
		buffer[bufferIndex] = reverb[i]
		bufferIndex = (bufferIndex + 1) % delaySamples
	}

	return reverb
}

// Delay applies a delay (echo) effect to the samples.
// delayTime: Delay time in seconds.
// feedback: Feedback factor (0.0 to less than 1.0).
// mix: Mixing proportion of the delayed signal (0.0 to 1.0).
func Delay(samples []float64, sampleRate int, delayTime float64, feedback float64, mix float64) []float64 {
	delaySamples := int(delayTime * float64(sampleRate))
	if delaySamples <= 0 {
		delaySamples = 1
	}
	delayed := make([]float64, len(samples))
	buffer := make([]float64, delaySamples)
	bufferIndex := 0

	for i, sample := range samples {
		delayedSample := buffer[bufferIndex]
		delayed[i] = sample*(1-mix) + delayedSample*mix
		buffer[bufferIndex] = sample + delayedSample*feedback
		bufferIndex = (bufferIndex + 1) % delaySamples
	}

	return delayed
}

// Compressor applies dynamic range compression to the samples.
// threshold: Threshold level above which compression is applied.
// ratio: Compression ratio.
// attack: Attack time in seconds.
// release: Release time in seconds.
// sampleRate: Sampling rate in Hz.
func Compressor(samples []float64, threshold, ratio, attack, release float64, sampleRate int) []float64 {
	compressed := make([]float64, len(samples))
	envelope := 0.0

	attackCoeff := math.Exp(-1.0 / (attack * float64(sampleRate)))
	releaseCoeff := math.Exp(-1.0 / (release * float64(sampleRate)))

	for i, sample := range samples {
		absSample := math.Abs(sample)

		if absSample > envelope {
			envelope = attackCoeff*envelope + (1.0-attackCoeff)*absSample
		} else {
			envelope = releaseCoeff*envelope + (1.0-releaseCoeff)*absSample
		}

		var gain float64
		if envelope > threshold {
			gain = threshold + (envelope-threshold)/ratio
			gain /= envelope
			if gain < 0.0 {
				gain = 0.0
			}
		} else {
			gain = 1.0
		}

		compressed[i] = sample * gain
	}

	return compressed
}

// Tremolo applies a tremolo effect by modulating the amplitude of the samples.
// rate: Modulation rate in Hz.
// depth: Modulation depth (0.0 to 1.0).
// sampleRate: Sampling rate in Hz.
func Tremolo(samples []float64, sampleRate int, rate, depth float64) []float64 {
	// Clamp depth to [0.0, 1.0]
	if depth < 0.0 {
		depth = 0.0
	} else if depth > 1.0 {
		depth = 1.0
	}

	tremolo := make([]float64, len(samples))
	phaseIncrement := rate / float64(sampleRate)
	phase := 0.0

	for i := range samples {
		modulation := 1.0 - depth + depth*math.Sin(2*math.Pi*phase)
		tremolo[i] = samples[i] * modulation
		phase += phaseIncrement
		if phase >= 1.0 {
			phase -= 1.0
		}
	}

	return tremolo
}

// Flanger applies a flanger effect to the samples.
// baseDelaySec: Base delay time in seconds.
// modDepthSec: Modulation depth in seconds.
// rate: Modulation rate in Hz.
// feedback: Feedback factor (0.0 to less than 1.0).
// mix: Mixing proportion of the delayed signal (0.0 to 1.0).
func Flanger(samples []float64, sampleRate int, baseDelaySec, modDepthSec, rate, feedback, mix float64) []float64 {
	baseDelaySamples := int(baseDelaySec * float64(sampleRate))
	modDepthSamples := int(modDepthSec * float64(sampleRate))
	maxDelay := baseDelaySamples + modDepthSamples
	if maxDelay <= 0 {
		maxDelay = 1
	}
	buffer := make([]float64, maxDelay)
	flanged := make([]float64, len(samples))
	bufferIndex := 0

	phaseIncrement := rate / float64(sampleRate)
	phase := 0.0

	for i, sample := range samples {
		// Calculate modulated delay
		modulation := math.Sin(2 * math.Pi * phase)
		currentDelay := baseDelaySamples + int(float64(modDepthSamples)*modulation)
		if currentDelay < 0 {
			currentDelay = 0
		} else if currentDelay >= maxDelay {
			currentDelay = maxDelay - 1
		}

		// Read delayed sample
		readIndex := (bufferIndex - currentDelay + maxDelay) % maxDelay
		delayedSample := buffer[readIndex]

		// Apply flanger effect
		flanged[i] = sample*(1-mix) + delayedSample*mix

		// Update buffer with current sample + feedback
		buffer[bufferIndex] = sample + delayedSample*feedback

		// Increment buffer index
		bufferIndex = (bufferIndex + 1) % maxDelay

		// Increment phase
		phase += phaseIncrement
		if phase >= 1.0 {
			phase -= 1.0
		}
	}

	// Clamp to avoid clipping
	for i := range flanged {
		if flanged[i] > 1.0 {
			flanged[i] = 1.0
		} else if flanged[i] < -1.0 {
			flanged[i] = -1.0
		}
	}

	return flanged
}

// Phaser applies a phaser effect to the samples.
// rate: Modulation rate in Hz.
// depth: Modulation depth.
// feedback: Feedback factor (0.0 to less than 1.0).
// sampleRate: Sampling rate in Hz.
func Phaser(samples []float64, sampleRate int, rate, depth, feedback float64) []float64 {
	phased := make([]float64, len(samples))
	phaseIncrement := rate / float64(sampleRate)
	phase := 0.0

	// Initialize two all-pass filters for a basic phaser
	// Note: A proper phaser would require multiple all-pass filters with phase shifts
	allPass1 := NewBiquad("all-pass", 1000.0, 0.7, float64(sampleRate))
	allPass2 := NewBiquad("all-pass", 1500.0, 0.7, float64(sampleRate))

	for i, sample := range samples {
		// Sweep the center frequency with LFO
		sweep := math.Sin(2 * math.Pi * phase)
		centerFreq := 1000.0 + depth*1000.0*sweep // Example frequency sweep from 0 to 2000 Hz
		if centerFreq < 20.0 {                    // Prevent frequencies below human hearing
			centerFreq = 20.0
		} else if centerFreq > float64(sampleRate)/2 {
			centerFreq = float64(sampleRate) / 2
		}

		// Update filter parameters dynamically
		allPass1.UpdateParameters("all-pass", centerFreq, 0.7, float64(sampleRate))
		allPass2.UpdateParameters("all-pass", centerFreq, 0.7, float64(sampleRate))

		// Apply all-pass filters with feedback
		out1 := allPass1.Process(sample + feedback*phased[i-1])
		out2 := allPass2.Process(out1 + feedback*phased[i-1])
		phased[i] = out2

		// Increment phase
		phase += phaseIncrement
		if phase >= 1.0 {
			phase -= 1.0
		}
	}

	return phased
}

// Bitcrusher reduces the bit depth and/or sample rate of the audio signal.
// bitDepth: Number of bits to retain (e.g., 8 for 8-bit).
// sampleRateReduction: Factor by which to reduce the sample rate (e.g., 2 to halve the sample rate).
func Bitcrusher(samples []float64, bitDepth int, sampleRateReduction int) []float64 {
	if bitDepth <= 0 {
		bitDepth = 16 // Default to 16 bits
	}
	if sampleRateReduction <= 0 {
		sampleRateReduction = 1 // No reduction
	}

	crushed := make([]float64, len(samples))
	step := math.Pow(2, float64(bitDepth))
	sampleCounter := 0
	var lastSample float64
	for i, sample := range samples {
		// Bit depth reduction
		quantized := math.Round(sample*step) / step

		// Sample rate reduction
		if sampleCounter%sampleRateReduction == 0 {
			lastSample = quantized
		}
		crushed[i] = lastSample
		sampleCounter++
	}
	return crushed
}

// RingModulation applies ring modulation to the samples using a carrier frequency.
// carrierFreq: Frequency of the carrier in Hz.
// sampleRate: Sampling rate in Hz.
func RingModulation(samples []float64, carrierFreq float64, sampleRate int) []float64 {
	modulated := make([]float64, len(samples))
	for i := range samples {
		t := float64(i) / float64(sampleRate)
		carrier := math.Sin(2 * math.Pi * carrierFreq * t)
		modulated[i] = samples[i] * carrier
	}
	return modulated
}

// WahWah applies a wah-wah effect to the samples.
// baseFreq: Base center frequency of the band-pass filter.
// sweepFreq: Frequency of the LFO sweeping the center frequency.
// q: Quality factor of the band-pass filter.
// sampleRate: Sampling rate in Hz.
func WahWah(samples []float64, sampleRate int, baseFreq, sweepFreq, q float64) []float64 {
	wah := make([]float64, len(samples))
	phaseIncrement := sweepFreq / float64(sampleRate)
	phase := 0.0

	// Initialize band-pass filter
	biquad := NewBiquad("band-pass", baseFreq, q, float64(sampleRate))

	for i, sample := range samples {
		// Sweep the center frequency with LFO
		sweep := math.Sin(2 * math.Pi * phase)
		centerFreq := baseFreq + sweep*baseFreq // Sweep from baseFreq - baseFreq to baseFreq + baseFreq
		if centerFreq < 20.0 {                  // Prevent frequencies below human hearing
			centerFreq = 20.0
		} else if centerFreq > float64(sampleRate)/2 {
			centerFreq = float64(sampleRate) / 2
		}

		// Update filter parameters
		biquad.UpdateParameters("band-pass", centerFreq, q, float64(sampleRate))

		// Apply band-pass filter
		wah[i] = biquad.Process(sample)

		// Increment phase
		phase += phaseIncrement
		if phase >= 1.0 {
			phase -= 1.0
		}
	}

	return wah
}

// StereoWidening enhances the stereo image by manipulating the amplitude differences between the left and right channels.
// width: Width factor (0.0 for mono, up to 1.0 for maximum widening).
func StereoWidening(left, right []float64, width float64) ([]float64, []float64) {
	widenedLeft := make([]float64, len(left))
	widenedRight := make([]float64, len(right))

	// Clamp width to [0.0, 1.0]
	if width < 0.0 {
		width = 0.0
	} else if width > 1.0 {
		width = 1.0
	}

	for i := range left {
		widenedLeft[i] = left[i] * (1 + width)
		widenedRight[i] = right[i] * (1 - width)
	}

	return widenedLeft, widenedRight
}

// NoiseGate suppresses audio signals that fall below a certain threshold.
// threshold: Threshold level below which the signal is attenuated.
// attack: Attack time in seconds.
// release: Release time in seconds.
// sampleRate: Sampling rate in Hz.
func NoiseGate(samples []float64, threshold, attack, release float64, sampleRate int) []float64 {
	gated := make([]float64, len(samples))
	gain := 1.0

	attackCoeff := math.Exp(-1.0 / (attack * float64(sampleRate)))
	releaseCoeff := math.Exp(-1.0 / (release * float64(sampleRate)))

	for i, sample := range samples {
		absSample := math.Abs(sample)

		if absSample > threshold {
			gain = attackCoeff*gain + (1.0-attackCoeff)*1.0
		} else {
			gain = releaseCoeff*gain + (1.0-releaseCoeff)*0.0
		}

		gated[i] = sample * gain
	}

	return gated
}

// StereoDelay implements separate delay effects on the left and right channels.
// delayTimeLeft: Delay time for the left channel in seconds.
// delayTimeRight: Delay time for the right channel in seconds.
// feedback: Feedback factor (0.0 to less than 1.0).
// mix: Mixing proportion of the delayed signal (0.0 to 1.0).
func StereoDelay(left, right []float64, sampleRate int, delayTimeLeft, delayTimeRight float64, feedback, mix float64) ([]float64, []float64) {
	delaySamplesLeft := int(delayTimeLeft * float64(sampleRate))
	delaySamplesRight := int(delayTimeRight * float64(sampleRate))
	if delaySamplesLeft <= 0 {
		delaySamplesLeft = 1
	}
	if delaySamplesRight <= 0 {
		delaySamplesRight = 1
	}
	delayedLeft := make([]float64, len(left))
	delayedRight := make([]float64, len(right))
	bufferLeft := make([]float64, delaySamplesLeft)
	bufferRight := make([]float64, delaySamplesRight)
	bufferIndexLeft := 0
	bufferIndexRight := 0

	for i := 0; i < len(left); i++ {
		// Left channel
		delayedSampleLeft := bufferLeft[bufferIndexLeft]
		delayedLeft[i] = left[i]*(1-mix) + delayedSampleLeft*mix
		bufferLeft[bufferIndexLeft] = left[i] + delayedSampleLeft*feedback
		bufferIndexLeft = (bufferIndexLeft + 1) % delaySamplesLeft

		// Right channel
		delayedSampleRight := bufferRight[bufferIndexRight]
		delayedRight[i] = right[i]*(1-mix) + delayedSampleRight*mix
		bufferRight[bufferIndexRight] = right[i] + delayedSampleRight*feedback
		bufferIndexRight = (bufferIndexRight + 1) % delaySamplesRight
	}

	return delayedLeft, delayedRight
}

// Expander increases the dynamic range by amplifying quieter signals and attenuating louder ones.
// threshold: Threshold level below which expansion is applied.
// ratio: Expansion ratio.
// attack: Attack time in seconds.
// release: Release time in seconds.
// sampleRate: Sampling rate in Hz.
func Expander(samples []float64, threshold, ratio, attack, release float64, sampleRate int) []float64 {
	expanded := make([]float64, len(samples))
	envelope := 0.0

	attackCoeff := math.Exp(-1.0 / (attack * float64(sampleRate)))
	releaseCoeff := math.Exp(-1.0 / (release * float64(sampleRate)))

	for i, sample := range samples {
		absSample := math.Abs(sample)

		if absSample > envelope {
			envelope = attackCoeff*envelope + (1.0-attackCoeff)*absSample
		} else {
			envelope = releaseCoeff*envelope + (1.0-releaseCoeff)*absSample
		}

		var gain float64
		if envelope < threshold {
			gain = envelope * ratio / threshold
			if gain > 1.0 {
				gain = 1.0
			}
		} else {
			gain = 1.0
		}

		expanded[i] = sample * gain
	}

	return expanded
}

// SoftClip applies a soft clipping distortion to a single sample.
// drive controls the amount of distortion.
func SoftClip(sample, drive float64) float64 {
	return (3 + drive) * sample / (1 + drive*math.Abs(sample))
}

// SoftClippingDistortion applies soft clipping distortion to the samples.
// drive controls the amount of distortion.
func SoftClippingDistortion(samples []float64, drive float64) []float64 {
	distorted := make([]float64, len(samples))
	for i, sample := range samples {
		distorted[i] = SoftClip(sample, drive)
		// Optional: Clamp to [-1, 1]
		if distorted[i] > 1.0 {
			distorted[i] = 1.0
		} else if distorted[i] < -1.0 {
			distorted[i] = -1.0
		}
	}
	return distorted
}

// SidechainCompressor applies compression to the target signal based on the trigger signal.
// target: The primary audio signal to be compressed.
// trigger: The secondary audio signal used to control the compression.
// threshold: Threshold level above which compression is applied.
// ratio: Compression ratio.
// attack: Attack time in seconds.
// release: Release time in seconds.
// sampleRate: Sampling rate in Hz.
func SidechainCompressor(target, trigger []float64, threshold, ratio, attack, release float64, sampleRate int) []float64 {
	if len(target) != len(trigger) {
		// Handle error: target and trigger must be the same length
		return target
	}

	compressed := make([]float64, len(target))
	gain := 1.0

	attackCoeff := math.Exp(-1.0 / (attack * float64(sampleRate)))
	releaseCoeff := math.Exp(-1.0 / (release * float64(sampleRate)))

	for i := 0; i < len(target); i++ {
		absTrigger := math.Abs(trigger[i])

		// Envelope follower
		if absTrigger > gain {
			gain = attackCoeff*gain + (1.0-attackCoeff)*absTrigger
		} else {
			gain = releaseCoeff*gain + (1.0-releaseCoeff)*absTrigger
		}

		// Gain computation based on threshold and ratio
		if gain > threshold {
			gain = threshold + (gain-threshold)/ratio
			gain /= gain // Normalize gain to prevent amplification
			if gain < 0.0 {
				gain = 0.0
			}
		} else {
			gain = 1.0
		}

		compressed[i] = target[i] * gain
	}

	return compressed
}

// MultibandCompression applies compression independently across different frequency bands.
// bands: Slice of frequency bands, each defined by Low and High cutoff frequencies.
// compressors: Slice of CompressorSettings structs corresponding to each band.
// sampleRate: Sampling rate in Hz.
func MultibandCompression(samples []float64, bands []struct {
	Low  float64
	High float64
}, compressors []CompressorSettings, sampleRate int) []float64 {
	if len(bands) != len(compressors) {
		// Handle error: bands and compressors must have the same length
		return samples
	}

	// Split into bands
	splitBands := make([][]float64, len(bands))
	for i, band := range bands {
		splitBands[i] = BandPassFilter(samples, band.Low, band.High, sampleRate)
	}

	// Compress each band
	for i := range splitBands {
		splitBands[i] = Compressor(splitBands[i], compressors[i].Threshold, compressors[i].Ratio, compressors[i].Attack, compressors[i].Release, sampleRate)
	}

	// Recombine bands
	recombined := make([]float64, len(samples))
	for _, band := range splitBands {
		for i := range recombined {
			recombined[i] += band[i]
		}
	}

	return recombined
}

// PitchShift shifts the pitch of the samples by the specified number of semitones.
// Note: This implementation uses simple resampling and changes the duration of the audio.
// For high-quality pitch shifting without altering duration, consider implementing advanced algorithms like Phase Vocoder or WSOLA.
func PitchShift(samples []float64, semitones float64) []float64 {
	rate := math.Pow(2, semitones/12)
	newLength := int(float64(len(samples)) / rate)
	shifted := make([]float64, newLength)
	for i := 0; i < newLength; i++ {
		srcIndex := float64(i) * rate
		lower := int(math.Floor(srcIndex))
		upper := lower + 1
		if upper >= len(samples) {
			upper = len(samples) - 1
		}
		frac := srcIndex - float64(lower)
		shifted[i] = samples[lower]*(1-frac) + samples[upper]*frac
	}
	return shifted
}

// audioeffects_test.go
package audioeffects

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

// Helper function to compare two float64 values within a specified tolerance.
func almostEqual(a, b float64, epsilon float64) bool {
	return math.Abs(a-b) <= epsilon
}

// Helper function to compare two slices of float64 values element-wise within a specified tolerance.
func slicesAlmostEqual(a, b []float64, epsilon float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !almostEqual(a[i], b[i], epsilon) {
			return false
		}
	}
	return true
}

// writeWav writes a slice of float64 samples to a WAV file.
// The samples are expected to be in the range [-1.0, 1.0].
func writeWav(filename string, samples []float64, sampleRate int) error {
	// Convert float64 samples to int16
	intSamples := make([]int, len(samples))
	for i, sample := range samples {
		// Clamp the sample to [-1.0, 1.0]
		if sample > 1.0 {
			sample = 1.0
		} else if sample < -1.0 {
			sample = -1.0
		}
		// Convert to int16
		intSamples[i] = int(math.Round(sample * 32767))
	}

	// Create an audio buffer
	buf := &audio.IntBuffer{
		Data:           intSamples,
		Format:         &audio.Format{SampleRate: sampleRate, NumChannels: 1},
		SourceBitDepth: 16,
	}

	// Create the WAV file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Create a WAV encoder
	enc := wav.NewEncoder(file, sampleRate, 16, 1, 1)

	// Write the buffer to the encoder
	if err := enc.Write(buf); err != nil {
		return err
	}

	// Close the encoder to flush the data
	if err := enc.Close(); err != nil {
		return err
	}

	return nil
}

// generateBassMelody generates a simple bass melody by mixing two sine waves.
// freq1 and freq2 are the frequencies of the oscillators.
func generateBassMelody(durationSeconds float64, sampleRate int, freq1, freq2 float64) []float64 {
	numSamples := int(durationSeconds * float64(sampleRate))
	samples := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		// Mix two sine waves
		samples[i] = 0.5*math.Sin(2*math.Pi*freq1*t) + 0.5*math.Sin(2*math.Pi*freq2*t)
		// Normalize to prevent clipping
		if samples[i] > 1.0 {
			samples[i] = 1.0
		} else if samples[i] < -1.0 {
			samples[i] = -1.0
		}
	}

	return samples
}

// TestEffectsGeneration generates a simple bass melody, applies various effects, and writes the results to WAV files.
func TestEffectsGeneration(t *testing.T) {
	sampleRate := 44100 // CD-quality sample rate
	duration := 2.0     // 2 seconds
	freq1 := 110.0      // A2 note
	freq2 := 220.0      // A3 note (octave above)

	// Create an output directory for WAV files
	outputDir := "test_outputs"
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		t.Fatalf("Failed to create output directory: %v", err)
	}

	// Step 1: Generate Bass Melody
	original := generateBassMelody(duration, sampleRate, freq1, freq2)

	// Write Original Melody to WAV
	originalPath := filepath.Join(outputDir, "original.wav")
	if err := writeWav(originalPath, original, sampleRate); err != nil {
		t.Fatalf("Failed to write original.wav: %v", err)
	}
	t.Logf("Generated %s", originalPath)

	// Step 2: Apply FadeIn
	fadedIn := FadeIn(append([]float64(nil), original...), 1.0, sampleRate) // 1-second fade-in
	fadeInPath := filepath.Join(outputDir, "fadein.wav")
	if err := writeWav(fadeInPath, fadedIn, sampleRate); err != nil {
		t.Fatalf("Failed to write fadein.wav: %v", err)
	}
	t.Logf("Generated %s", fadeInPath)

	// Step 3: Apply LowPassFilter
	lowPassed := LowPassFilter(append([]float64(nil), original...), 1000.0, sampleRate) // 1kHz cutoff
	lowPassPath := filepath.Join(outputDir, "lowpass.wav")
	if err := writeWav(lowPassPath, lowPassed, sampleRate); err != nil {
		t.Fatalf("Failed to write lowpass.wav: %v", err)
	}
	t.Logf("Generated %s", lowPassPath)

	// Step 4: Apply HighPassFilter
	highPassed := HighPassFilter(append([]float64(nil), original...), 500.0, sampleRate) // 500Hz cutoff
	highPassPath := filepath.Join(outputDir, "highpass.wav")
	if err := writeWav(highPassPath, highPassed, sampleRate); err != nil {
		t.Fatalf("Failed to write highpass.wav: %v", err)
	}
	t.Logf("Generated %s", highPassPath)

	// Step 5: Apply BandPassFilter
	bandPassed := BandPassFilter(append([]float64(nil), original...), 500.0, 1000.0, sampleRate) // 500Hz-1kHz band
	bandPassPath := filepath.Join(outputDir, "bandpass.wav")
	if err := writeWav(bandPassPath, bandPassed, sampleRate); err != nil {
		t.Fatalf("Failed to write bandpass.wav: %v", err)
	}
	t.Logf("Generated %s", bandPassPath)

	// Step 6: Apply Drive (Distortion)
	// Apply Drive to each sample individually
	driven := make([]float64, len(original))
	for i, sample := range original {
		driven[i] = Drive(sample, 0.7) // 70% drive
	}
	drivePath := filepath.Join(outputDir, "drive.wav")
	if err := writeWav(drivePath, driven, sampleRate); err != nil {
		t.Fatalf("Failed to write drive.wav: %v", err)
	}
	t.Logf("Generated %s", drivePath)

	// Step 7: Apply Chorus
	chorused := Chorus(append([]float64(nil), original...), sampleRate, 0.05, 0.02, 1.5, 0.3) // 50ms delay, 2% depth, 1.5Hz rate, 30% mix
	chorusPath := filepath.Join(outputDir, "chorus.wav")
	if err := writeWav(chorusPath, chorused, sampleRate); err != nil {
		t.Fatalf("Failed to write chorus.wav: %v", err)
	}
	t.Logf("Generated %s", chorusPath)

	// Step 8: Apply Schroeder Reverb
	// Updated delay times: Using longer delays for natural reverb
	combDelays := []int{
		int(0.050 * float64(sampleRate)), // 50ms
		int(0.056 * float64(sampleRate)), // 56ms
		int(0.062 * float64(sampleRate)), // 62ms
		int(0.068 * float64(sampleRate)), // 68ms
	} // [50ms, 56ms, 62ms, 68ms] at 44100 Hz
	allPassDelays := []int{
		int(0.005 * float64(sampleRate)), // 5ms
		int(0.007 * float64(sampleRate)), // 7ms
	} // [5ms, 7ms] at 44100 Hz
	reverb, err := SchroederReverb(append([]float64(nil), original...), 0.6, combDelays, allPassDelays)
	if err != nil {
		t.Fatalf("SchroederReverb error: %v", err)
	}
	reverbPath := filepath.Join(outputDir, "reverb.wav")
	if err := writeWav(reverbPath, reverb, sampleRate); err != nil {
		t.Fatalf("Failed to write reverb.wav: %v", err)
	}
	t.Logf("Generated %s", reverbPath)

	// Step 9: Apply Dark Reverb
	darkReverb, err := DarkReverb(append([]float64(nil), original...), 0.6, combDelays, allPassDelays, 300.0, sampleRate)
	if err != nil {
		t.Fatalf("DarkReverb error: %v", err)
	}
	darkReverbPath := filepath.Join(outputDir, "dark_reverb.wav")
	if err := writeWav(darkReverbPath, darkReverb, sampleRate); err != nil {
		t.Fatalf("Failed to write dark_reverb.wav: %v", err)
	}
	t.Logf("Generated %s", darkReverbPath)

	// Step 10: Apply Limiter
	limited := Limiter(append([]float64(nil), original...))
	limiterPath := filepath.Join(outputDir, "limited.wav")
	if err := writeWav(limiterPath, limited, sampleRate); err != nil {
		t.Fatalf("Failed to write limited.wav: %v", err)
	}
	t.Logf("Generated %s", limiterPath)

	// Step 11: Apply NormalizeSamples
	normalized := NormalizeSamples(append([]float64(nil), original...), 0.8)
	normalizedPath := filepath.Join(outputDir, "normalized.wav")
	if err := writeWav(normalizedPath, normalized, sampleRate); err != nil {
		t.Fatalf("Failed to write normalized.wav: %v", err)
	}
	t.Logf("Generated %s", normalizedPath)
}

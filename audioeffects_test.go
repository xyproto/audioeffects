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
// numChannels specifies the number of audio channels (1 for mono, 2 for stereo).
func writeWav(filename string, samples []float64, sampleRate int, numChannels int) error {
	// Convert float64 samples to int16
	intSamples := make([]int, len(samples)*numChannels)
	for i, sample := range samples {
		// Clamp the sample to [-1.0, 1.0]
		if sample > 1.0 {
			sample = 1.0
		} else if sample < -1.0 {
			sample = -1.0
		}
		// Convert to int16
		intSample := int(math.Round(sample * 32767))
		for ch := 0; ch < numChannels; ch++ {
			intSamples[i*numChannels+ch] = intSample
		}
	}

	// Create an audio buffer
	buf := &audio.IntBuffer{
		Data:           intSamples,
		Format:         &audio.Format{SampleRate: sampleRate, NumChannels: numChannels},
		SourceBitDepth: 16,
	}

	// Create the WAV file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Create a WAV encoder
	enc := wav.NewEncoder(file, sampleRate, 16, numChannels, 1)

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
	if err := writeWav(originalPath, original, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write original.wav: %v", err)
	}
	t.Logf("Generated %s", originalPath)

	// Step 2: Apply FadeIn
	fadedIn := FadeIn(append([]float64(nil), original...), 1.0, sampleRate) // 1-second fade-in
	fadeInPath := filepath.Join(outputDir, "fadein.wav")
	if err := writeWav(fadeInPath, fadedIn, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write fadein.wav: %v", err)
	}
	t.Logf("Generated %s", fadeInPath)

	// Step 3: Apply FadeOut
	fadedOut := FadeOut(append([]float64(nil), original...), 1.0, sampleRate) // 1-second fade-out
	fadeOutPath := filepath.Join(outputDir, "fadeout.wav")
	if err := writeWav(fadeOutPath, fadedOut, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write fadeout.wav: %v", err)
	}
	t.Logf("Generated %s", fadeOutPath)

	// Step 4: Apply LowPassFilter
	lowPassed := LowPassFilter(append([]float64(nil), original...), 1000.0, sampleRate) // 1kHz cutoff
	lowPassPath := filepath.Join(outputDir, "lowpass.wav")
	if err := writeWav(lowPassPath, lowPassed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write lowpass.wav: %v", err)
	}
	t.Logf("Generated %s", lowPassPath)

	// Step 5: Apply HighPassFilter
	highPassed := HighPassFilter(append([]float64(nil), original...), 500.0, sampleRate) // 500Hz cutoff
	highPassPath := filepath.Join(outputDir, "highpass.wav")
	if err := writeWav(highPassPath, highPassed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write highpass.wav: %v", err)
	}
	t.Logf("Generated %s", highPassPath)

	// Step 6: Apply BandPassFilter
	bandPassed := BandPassFilter(append([]float64(nil), original...), 500.0, 1000.0, sampleRate) // 500Hz-1kHz band
	bandPassPath := filepath.Join(outputDir, "bandpass.wav")
	if err := writeWav(bandPassPath, bandPassed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write bandpass.wav: %v", err)
	}
	t.Logf("Generated %s", bandPassPath)

	// Step 7: Apply Drive (Distortion)
	// Apply Drive to each sample individually
	driven := make([]float64, len(original))
	for i, sample := range original {
		driven[i] = Drive(sample, 0.7) // 70% drive
	}
	drivePath := filepath.Join(outputDir, "drive.wav")
	if err := writeWav(drivePath, driven, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write drive.wav: %v", err)
	}
	t.Logf("Generated %s", drivePath)

	// Step 8: Apply Chorus
	chorused := Chorus(append([]float64(nil), original...), sampleRate, 0.05, 0.02, 1.5, 0.3) // 50ms delay, 2% depth, 1.5Hz rate, 30% mix
	chorusPath := filepath.Join(outputDir, "chorus.wav")
	if err := writeWav(chorusPath, chorused, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write chorus.wav: %v", err)
	}
	t.Logf("Generated %s", chorusPath)

	// Step 9: Apply Reverb
	// Define multiple delay times and corresponding decays
	delayTimes := []float64{0.02, 0.045, 0.065, 0.085} // 20ms, 45ms, 65ms, 85ms
	decays := []float64{0.5, 0.4, 0.3, 0.2}            // Decay factors for each delay line
	reverbed := Reverb(append([]float64(nil), original...), sampleRate, delayTimes, decays)
	reverbPath := filepath.Join(outputDir, "reverb.wav")
	if err := writeWav(reverbPath, reverbed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write reverb.wav: %v", err)
	}
	t.Logf("Generated %s", reverbPath)

	// Step 10: Apply Limiter
	limited := Limiter(append([]float64(nil), original...))
	limiterPath := filepath.Join(outputDir, "limited.wav")
	if err := writeWav(limiterPath, limited, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write limited.wav: %v", err)
	}
	t.Logf("Generated %s", limiterPath)

	// Step 11: Apply NormalizeSamples
	normalized := NormalizeSamples(append([]float64(nil), original...), 0.8)
	normalizedPath := filepath.Join(outputDir, "normalized.wav")
	if err := writeWav(normalizedPath, normalized, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write normalized.wav: %v", err)
	}
	t.Logf("Generated %s", normalizedPath)

	// Step 12: Apply Compressor
	compressed := Compressor(append([]float64(nil), original...), -0.5, 4.0, 0.01, 0.1, sampleRate) // Threshold: -0.5, Ratio: 4:1, Attack: 10ms, Release: 100ms
	compressorPath := filepath.Join(outputDir, "compressor.wav")
	if err := writeWav(compressorPath, compressed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write compressor.wav: %v", err)
	}
	t.Logf("Generated %s", compressorPath)

	// Step 13: Apply Envelope
	adsr := Envelope(append([]float64(nil), original...), 0.2, 0.2, 0.7, 0.5, sampleRate) // Attack: 200ms, Decay: 200ms, Sustain: 70%, Release: 500ms
	envelopedPath := filepath.Join(outputDir, "envelope.wav")
	if err := writeWav(envelopedPath, adsr, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write envelope.wav: %v", err)
	}
	t.Logf("Generated %s", envelopedPath)

	// Step 14: Apply Panning
	left, right := Panning(append([]float64(nil), original...), 0.5) // Pan 50% to the right
	pannedPathLeft := filepath.Join(outputDir, "panned_left.wav")
	pannedPathRight := filepath.Join(outputDir, "panned_right.wav")
	if err := writeWav(pannedPathLeft, left, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write panned_left.wav: %v", err)
	}
	t.Logf("Generated %s", pannedPathLeft)
	if err := writeWav(pannedPathRight, right, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write panned_right.wav: %v", err)
	}
	t.Logf("Generated %s", pannedPathRight)

	// Step 15: Apply FrequencyModulation
	// Set carrier frequency to 440Hz (A4) and modulation depth to 100Hz
	frequencyModulated := FrequencyModulation(append([]float64(nil), original...), 440.0, 100.0, float64(sampleRate))
	frequencyModPath := filepath.Join(outputDir, "frequencymodulation.wav")
	if err := writeWav(frequencyModPath, frequencyModulated, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write frequencymodulation.wav: %v", err)
	}
	t.Logf("Generated %s", frequencyModPath)

	// Step 16: Apply PitchModulation
	// Set modulation frequency to 5Hz and depth to 0.005 seconds (~5ms vibrato)
	pitchModulated := PitchModulation(append([]float64(nil), original...), 5.0, 0.005, sampleRate)
	pitchModPath := filepath.Join(outputDir, "pitchmodulation.wav")
	if err := writeWav(pitchModPath, pitchModulated, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write pitchmodulation.wav: %v", err)
	}
	t.Logf("Generated %s", pitchModPath)

	// Step 17: Apply Tremolo
	tremoloed := Tremolo(append([]float64(nil), original...), sampleRate, 5.0, 0.7) // 5Hz rate, 70% depth
	tremoloPath := filepath.Join(outputDir, "tremolo.wav")
	if err := writeWav(tremoloPath, tremoloed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write tremolo.wav: %v", err)
	}
	t.Logf("Generated %s", tremoloPath)

	// Step 18: Apply Flanger
	flanged := Flanger(append([]float64(nil), original...), sampleRate, 0.002, 0.001, 0.25, 0.5, 0.5) // 2ms base delay, 1ms mod depth, 0.25Hz rate, 50% feedback, 50% mix
	flangerPath := filepath.Join(outputDir, "flanger.wav")
	if err := writeWav(flangerPath, flanged, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write flanger.wav: %v", err)
	}
	t.Logf("Generated %s", flangerPath)

	// Step 19: Apply Phaser
	phased := Phaser(append([]float64(nil), original...), sampleRate, 0.5, 500.0, 0.7) // 0.5Hz rate, 500 depth, 70% feedback
	phaserPath := filepath.Join(outputDir, "phaser.wav")
	if err := writeWav(phaserPath, phased, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write phaser.wav: %v", err)
	}
	t.Logf("Generated %s", phaserPath)

	// Step 20: Apply Bitcrusher
	bitcrushed := Bitcrusher(append([]float64(nil), original...), 8, 2) // 8-bit depth, reduce sample rate by factor of 2
	bitcrusherPath := filepath.Join(outputDir, "bitcrusher.wav")
	if err := writeWav(bitcrusherPath, bitcrushed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write bitcrusher.wav: %v", err)
	}
	t.Logf("Generated %s", bitcrusherPath)

	// Step 21: Apply RingModulation
	ringModulated := RingModulation(append([]float64(nil), original...), 30.0, sampleRate) // 30Hz carrier frequency
	ringModPath := filepath.Join(outputDir, "ringmodulation.wav")
	if err := writeWav(ringModPath, ringModulated, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write ringmodulation.wav: %v", err)
	}
	t.Logf("Generated %s", ringModPath)

	// Step 22: Apply WahWah
	wahWah := WahWah(append([]float64(nil), original...), sampleRate, 800.0, 2.0, 0.7) // Base freq: 800Hz, Sweep freq: 2Hz, Q: 0.7
	wahWahPath := filepath.Join(outputDir, "wahwah.wav")
	if err := writeWav(wahWahPath, wahWah, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write wahwah.wav: %v", err)
	}
	t.Logf("Generated %s", wahWahPath)

	// Step 23: Apply StereoWidening
	// First, create a stereo signal by duplicating the original
	leftChannel, rightChannel := Panning(append([]float64(nil), original...), 0.0) // Start with centered panning
	widenedLeft, widenedRight := StereoWidening(leftChannel, rightChannel, 0.5)    // 50% widening
	// Combine widened channels into interleaved stereo samples
	stereoSamples := make([]float64, len(widenedLeft)*2)
	for i := 0; i < len(widenedLeft); i++ {
		stereoSamples[2*i] = widenedLeft[i]
		stereoSamples[2*i+1] = widenedRight[i]
	}
	stereoWideningPath := filepath.Join(outputDir, "stereowidening.wav")
	if err := writeWav(stereoWideningPath, stereoSamples, sampleRate, 2); err != nil {
		t.Fatalf("Failed to write stereowidening.wav: %v", err)
	}
	t.Logf("Generated %s", stereoWideningPath)

	// Step 24: Apply NoiseGate
	gated := NoiseGate(append([]float64(nil), original...), 0.3, 0.01, 0.1, sampleRate) // Threshold: 0.3, Attack: 10ms, Release: 100ms
	noiseGatePath := filepath.Join(outputDir, "noisegate.wav")
	if err := writeWav(noiseGatePath, gated, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write noisegate.wav: %v", err)
	}
	t.Logf("Generated %s", noiseGatePath)

	// Step 25: Apply StereoDelay
	// Create stereo channels
	left, right = Panning(append([]float64(nil), original...), -0.5)                      // Pan 50% to the left
	delayedLeft, delayedRight := StereoDelay(left, right, sampleRate, 0.3, 0.4, 0.5, 0.5) // 300ms left, 400ms right, 50% feedback, 50% mix
	// Combine delayed channels into interleaved stereo samples
	stereoDelayed := make([]float64, len(delayedLeft)*2)
	for i := 0; i < len(delayedLeft); i++ {
		stereoDelayed[2*i] = delayedLeft[i]
		stereoDelayed[2*i+1] = delayedRight[i]
	}
	stereoDelayPath := filepath.Join(outputDir, "stereodelay.wav")
	if err := writeWav(stereoDelayPath, stereoDelayed, sampleRate, 2); err != nil {
		t.Fatalf("Failed to write stereodelay.wav: %v", err)
	}
	t.Logf("Generated %s", stereoDelayPath)

	// Step 26: Apply Expander
	expanded := Expander(append([]float64(nil), original...), -0.5, 2.0, 0.01, 0.1, sampleRate) // Threshold: -0.5, Ratio: 2:1, Attack: 10ms, Release: 100ms
	expanderPath := filepath.Join(outputDir, "expander.wav")
	if err := writeWav(expanderPath, expanded, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write expander.wav: %v", err)
	}
	t.Logf("Generated %s", expanderPath)

	// Step 27: Apply SoftClippingDistortion
	softClipped := SoftClippingDistortion(append([]float64(nil), original...), 2.0) // Drive: 2.0
	softClipPath := filepath.Join(outputDir, "softclip.wav")
	if err := writeWav(softClipPath, softClipped, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write softclip.wav: %v", err)
	}
	t.Logf("Generated %s", softClipPath)

	// Step 28: Apply SidechainCompressor
	// For simplicity, use the original signal as both target and trigger
	sidechained := SidechainCompressor(append([]float64(nil), original...), append([]float64(nil), original...), -0.5, 4.0, 0.01, 0.1, sampleRate)
	sidechainPath := filepath.Join(outputDir, "sidechaincompressor.wav")
	if err := writeWav(sidechainPath, sidechained, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write sidechaincompressor.wav: %v", err)
	}
	t.Logf("Generated %s", sidechainPath)

	// Step 29: Apply MultibandCompression
	bands := []struct {
		Low  float64
		High float64
	}{
		{Low: 20.0, High: 200.0},     // Bass
		{Low: 200.0, High: 2000.0},   // Mid
		{Low: 2000.0, High: 20000.0}, // Treble
	}
	compressors := []CompressorSettings{
		{Threshold: -0.5, Ratio: 3.0, Attack: 0.01, Release: 0.1},
		{Threshold: -0.5, Ratio: 3.0, Attack: 0.01, Release: 0.1},
		{Threshold: -0.5, Ratio: 3.0, Attack: 0.01, Release: 0.1},
	}
	multibandCompressed := MultibandCompression(append([]float64(nil), original...), bands, compressors, sampleRate)
	multibandPath := filepath.Join(outputDir, "multibandcompression.wav")
	if err := writeWav(multibandPath, multibandCompressed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write multibandcompression.wav: %v", err)
	}
	t.Logf("Generated %s", multibandPath)

	// Step 30: Apply PitchShift
	pitchShifted := PitchShift(append([]float64(nil), original...), 5.0) // Shift up by 5 semitones
	pitchShiftPath := filepath.Join(outputDir, "pitchshift.wav")
	if err := writeWav(pitchShiftPath, pitchShifted, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write pitchshift.wav: %v", err)
	}
	t.Logf("Generated %s", pitchShiftPath)
}

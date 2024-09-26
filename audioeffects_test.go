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

func almostEqual(a, b float64, epsilon float64) bool {
	return math.Abs(a-b) <= epsilon
}

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

func writeWav(filename string, samples []float64, sampleRate int, numChannels int) error {
	intSamples := make([]int, len(samples)*numChannels)
	for i, sample := range samples {
		if sample > 1.0 {
			sample = 1.0
		} else if sample < -1.0 {
			sample = -1.0
		}
		intSample := int(math.Round(sample * 32767))
		for ch := 0; ch < numChannels; ch++ {
			intSamples[i*numChannels+ch] = intSample
		}
	}

	buf := &audio.IntBuffer{
		Data:           intSamples,
		Format:         &audio.Format{SampleRate: sampleRate, NumChannels: numChannels},
		SourceBitDepth: 16,
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	enc := wav.NewEncoder(file, sampleRate, 16, numChannels, 1)
	if err := enc.Write(buf); err != nil {
		return err
	}

	if err := enc.Close(); err != nil {
		return err
	}

	return nil
}

func generateBassMelody(durationSeconds float64, sampleRate int, freq1, freq2 float64) []float64 {
	numSamples := int(durationSeconds * float64(sampleRate))
	samples := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		samples[i] = 0.5*math.Sin(2*math.Pi*freq1*t) + 0.5*math.Sin(2*math.Pi*freq2*t)
		if samples[i] > 1.0 {
			samples[i] = 1.0
		} else if samples[i] < -1.0 {
			samples[i] = -1.0
		}
	}

	return samples
}

func TestEffectsGeneration(t *testing.T) {
	sampleRate := 44100
	duration := 2.0
	freq1 := 110.0
	freq2 := 220.0

	outputDir := "test_outputs"
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		t.Fatalf("Failed to create output directory: %v", err)
	}

	original := generateBassMelody(duration, sampleRate, freq1, freq2)
	originalPath := filepath.Join(outputDir, "original.wav")
	if err := writeWav(originalPath, original, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write original.wav: %v", err)
	}
	t.Logf("Generated %s", originalPath)

	fadedIn := FadeIn(append([]float64(nil), original...), 1.0, sampleRate)
	fadeInPath := filepath.Join(outputDir, "fadein.wav")
	if err := writeWav(fadeInPath, fadedIn, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write fadein.wav: %v", err)
	}
	t.Logf("Generated %s", fadeInPath)

	fadedOut := FadeOut(append([]float64(nil), original...), 1.0, sampleRate)
	fadeOutPath := filepath.Join(outputDir, "fadeout.wav")
	if err := writeWav(fadeOutPath, fadedOut, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write fadeout.wav: %v", err)
	}
	t.Logf("Generated %s", fadeOutPath)

	lowPassed := LowPassFilter(append([]float64(nil), original...), 1000.0, sampleRate)
	lowPassPath := filepath.Join(outputDir, "lowpass.wav")
	if err := writeWav(lowPassPath, lowPassed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write lowpass.wav: %v", err)
	}
	t.Logf("Generated %s", lowPassPath)

	highPassed := HighPassFilter(append([]float64(nil), original...), 500.0, sampleRate)
	highPassPath := filepath.Join(outputDir, "highpass.wav")
	if err := writeWav(highPassPath, highPassed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write highpass.wav: %v", err)
	}
	t.Logf("Generated %s", highPassPath)

	bandPassed := BandPassFilter(append([]float64(nil), original...), 500.0, 1000.0, sampleRate)
	bandPassPath := filepath.Join(outputDir, "bandpass.wav")
	if err := writeWav(bandPassPath, bandPassed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write bandpass.wav: %v", err)
	}
	t.Logf("Generated %s", bandPassPath)

	driven := make([]float64, len(original))
	for i, sample := range original {
		driven[i] = Drive(sample, 0.7)
	}
	drivePath := filepath.Join(outputDir, "drive.wav")
	if err := writeWav(drivePath, driven, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write drive.wav: %v", err)
	}
	t.Logf("Generated %s", drivePath)

	chorused := Chorus(append([]float64(nil), original...), sampleRate, 0.05, 0.02, 1.5, 0.3)
	chorusPath := filepath.Join(outputDir, "chorus.wav")
	if err := writeWav(chorusPath, chorused, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write chorus.wav: %v", err)
	}
	t.Logf("Generated %s", chorusPath)

	delayTimes := []float64{0.02, 0.045, 0.065, 0.085}
	decays := []float64{0.5, 0.4, 0.3, 0.2}
	mix := 0.5
	reverbed := Reverb(append([]float64(nil), original...), sampleRate, delayTimes, decays, mix)
	reverbPath := filepath.Join(outputDir, "reverb.wav")
	if err := writeWav(reverbPath, reverbed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write reverb.wav: %v", err)
	}
	t.Logf("Generated %s", reverbPath)

	limited := Limiter(append([]float64(nil), original...))
	limiterPath := filepath.Join(outputDir, "limited.wav")
	if err := writeWav(limiterPath, limited, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write limited.wav: %v", err)
	}
	t.Logf("Generated %s", limiterPath)

	normalized := NormalizeSamples(append([]float64(nil), original...), 0.8)
	normalizedPath := filepath.Join(outputDir, "normalized.wav")
	if err := writeWav(normalizedPath, normalized, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write normalized.wav: %v", err)
	}
	t.Logf("Generated %s", normalizedPath)

	compressed := Compressor(append([]float64(nil), original...), -0.5, 4.0, 0.01, 0.1, sampleRate)
	compressorPath := filepath.Join(outputDir, "compressor.wav")
	if err := writeWav(compressorPath, compressed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write compressor.wav: %v", err)
	}
	t.Logf("Generated %s", compressorPath)

	adsr := Envelope(append([]float64(nil), original...), 0.2, 0.2, 0.7, 0.5, sampleRate)
	envelopedPath := filepath.Join(outputDir, "envelope.wav")
	if err := writeWav(envelopedPath, adsr, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write envelope.wav: %v", err)
	}
	t.Logf("Generated %s", envelopedPath)

	left, right := Panning(append([]float64(nil), original...), 0.5)
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

	frequencyModulated := FrequencyModulation(append([]float64(nil), original...), 440.0, 100.0, float64(sampleRate))
	frequencyModPath := filepath.Join(outputDir, "frequencymodulation.wav")
	if err := writeWav(frequencyModPath, frequencyModulated, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write frequencymodulation.wav: %v", err)
	}
	t.Logf("Generated %s", frequencyModPath)

	pitchModulated := PitchModulation(append([]float64(nil), original...), 5.0, 0.005, sampleRate)
	pitchModPath := filepath.Join(outputDir, "pitchmodulation.wav")
	if err := writeWav(pitchModPath, pitchModulated, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write pitchmodulation.wav: %v", err)
	}
	t.Logf("Generated %s", pitchModPath)

	tremoloed := Tremolo(append([]float64(nil), original...), sampleRate, 5.0, 0.7)
	tremoloPath := filepath.Join(outputDir, "tremolo.wav")
	if err := writeWav(tremoloPath, tremoloed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write tremolo.wav: %v", err)
	}
	t.Logf("Generated %s", tremoloPath)

	flanged := Flanger(append([]float64(nil), original...), sampleRate, 0.002, 0.001, 0.25, 0.5, 0.5)
	flangerPath := filepath.Join(outputDir, "flanger.wav")
	if err := writeWav(flangerPath, flanged, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write flanger.wav: %v", err)
	}
	t.Logf("Generated %s", flangerPath)

	phased := Phaser(append([]float64(nil), original...), sampleRate, 0.5, 500.0, 0.7)
	phaserPath := filepath.Join(outputDir, "phaser.wav")
	if err := writeWav(phaserPath, phased, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write phaser.wav: %v", err)
	}
	t.Logf("Generated %s", phaserPath)

	bitcrushed := Bitcrusher(append([]float64(nil), original...), 8, 2)
	bitcrusherPath := filepath.Join(outputDir, "bitcrusher.wav")
	if err := writeWav(bitcrusherPath, bitcrushed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write bitcrusher.wav: %v", err)
	}
	t.Logf("Generated %s", bitcrusherPath)

	ringModulated := RingModulation(append([]float64(nil), original...), 30.0, sampleRate)
	ringModPath := filepath.Join(outputDir, "ringmodulation.wav")
	if err := writeWav(ringModPath, ringModulated, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write ringmodulation.wav: %v", err)
	}
	t.Logf("Generated %s", ringModPath)

	wahWah := WahWah(append([]float64(nil), original...), sampleRate, 800.0, 2.0, 0.7)
	wahWahPath := filepath.Join(outputDir, "wahwah.wav")
	if err := writeWav(wahWahPath, wahWah, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write wahwah.wav: %v", err)
	}
	t.Logf("Generated %s", wahWahPath)

	// Step 23: Apply StereoWidening
	leftChannel, rightChannel := Panning(append([]float64(nil), original...), 0.0)
	widenedLeft, widenedRight := StereoWidening(leftChannel, rightChannel, 0.5)
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
	gated := NoiseGate(append([]float64(nil), original...), 0.3, 0.01, 0.1, sampleRate)
	noiseGatePath := filepath.Join(outputDir, "noisegate.wav")
	if err := writeWav(noiseGatePath, gated, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write noisegate.wav: %v", err)
	}
	t.Logf("Generated %s", noiseGatePath)

	// Step 25: Apply StereoDelay
	left, right = Panning(append([]float64(nil), original...), -0.5)
	delayedLeft, delayedRight := StereoDelay(left, right, sampleRate, 0.3, 0.4, 0.5, 0.5)
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
	expanded := Expander(append([]float64(nil), original...), -0.5, 2.0, 0.01, 0.1, sampleRate)
	expanderPath := filepath.Join(outputDir, "expander.wav")
	if err := writeWav(expanderPath, expanded, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write expander.wav: %v", err)
	}
	t.Logf("Generated %s", expanderPath)

	// Step 27: Apply SoftClippingDistortion
	softClipped := SoftClippingDistortion(append([]float64(nil), original...), 2.0)
	softClipPath := filepath.Join(outputDir, "softclip.wav")
	if err := writeWav(softClipPath, softClipped, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write softclip.wav: %v", err)
	}
	t.Logf("Generated %s", softClipPath)

	// Step 28: Apply SidechainCompressor
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
		{Low: 20.0, High: 200.0},
		{Low: 200.0, High: 2000.0},
		{Low: 2000.0, High: 20000.0},
	}
	compressors := []struct {
		Threshold float64
		Ratio     float64
		Attack    float64
		Release   float64
	}{
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
	pitchShifted := PitchShift(append([]float64(nil), original...), 5.0)
	pitchShiftPath := filepath.Join(outputDir, "pitchshift.wav")
	if err := writeWav(pitchShiftPath, pitchShifted, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write pitchshift.wav: %v", err)
	}
	t.Logf("Generated %s", pitchShiftPath)
}

// TestQuadraticFadeIn generates a WAV file with QuadraticFadeIn applied for manual inspection.
func TestQuadraticFadeIn(t *testing.T) {
	sampleRate := 44100
	duration := 1.0 // seconds
	fadeDuration := 0.5
	freq := 440.0

	outputDir := "test_outputs"
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		t.Fatalf("Failed to create output directory: %v", err)
	}

	numSamples := int(duration * float64(sampleRate))
	samples := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		tVal := float64(i) / float64(sampleRate)
		samples[i] = math.Sin(2 * math.Pi * freq * tVal)
	}

	fadedIn := QuadraticFadeIn(samples, fadeDuration, sampleRate)
	fadeInPath := filepath.Join(outputDir, "quadratic_fadein.wav")
	if err := writeWav(fadeInPath, fadedIn, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write quadratic_fadein.wav: %v", err)
	}
	t.Logf("Generated %s", fadeInPath)
}

// TestQuadraticFadeOut generates a WAV file with QuadraticFadeOut applied for manual inspection.
func TestQuadraticFadeOut(t *testing.T) {
	sampleRate := 44100
	duration := 1.0 // seconds
	fadeDuration := 0.5
	freq := 440.0

	outputDir := "test_outputs"
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		t.Fatalf("Failed to create output directory: %v", err)
	}

	numSamples := int(duration * float64(sampleRate))
	samples := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		tVal := float64(i) / float64(sampleRate)
		samples[i] = math.Sin(2 * math.Pi * freq * tVal)
	}

	fadedOut := QuadraticFadeOut(samples, fadeDuration, sampleRate)
	fadeOutPath := filepath.Join(outputDir, "quadratic_fadeout.wav")
	if err := writeWav(fadeOutPath, fadedOut, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write quadratic_fadeout.wav: %v", err)
	}
	t.Logf("Generated %s", fadeOutPath)
}

// TestEnvelopeAtTime generates a WAV file representing the ADSR envelope for manual inspection.
func TestEnvelopeAtTime(t *testing.T) {
	sampleRate := 44100
	duration := 2.0 // seconds
	attack := 0.5
	decay := 0.3
	sustainLevel := 0.7
	release := 0.5

	outputDir := "test_outputs"
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		t.Fatalf("Failed to create output directory: %v", err)
	}

	numSamples := int(duration * float64(sampleRate))
	samples := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		tVal := float64(i) / float64(sampleRate)
		envelope := EnvelopeAtTime(tVal, attack, decay, sustainLevel, release, duration)
		samples[i] = envelope
	}

	envelopePath := filepath.Join(outputDir, "envelope_at_time.wav")
	if err := writeWav(envelopePath, samples, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write envelope_at_time.wav: %v", err)
	}
	t.Logf("Generated %s", envelopePath)
}

// TestShimmer applies the Shimmer effect and generates a WAV file for manual inspection.
func TestShimmer(t *testing.T) {
	sampleRate := 44100
	duration := 2.0 // seconds
	freq1 := 110.0
	freq2 := 220.0

	outputDir := "test_outputs"
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		t.Fatalf("Failed to create output directory: %v", err)
	}

	original := generateBassMelody(duration, sampleRate, freq1, freq2)

	// Apply Shimmer
	shimmered := Shimmer(original, sampleRate, 0.3, 0.5, 12.0, 0.3) // 12 semitones, 30% feedback
	shimmerPath := filepath.Join(outputDir, "shimmer.wav")
	if err := writeWav(shimmerPath, shimmered, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write shimmer.wav: %v", err)
	}
	t.Logf("Generated %s", shimmerPath)
}

// TestShimmerBitcrusher applies the ShimmerBitcrusher effect with feedback and generates a WAV file for manual inspection.
func TestShimmerBitcrusher(t *testing.T) {
	sampleRate := 44100
	duration := 2.0 // seconds
	freq1 := 110.0
	freq2 := 220.0

	outputDir := "test_outputs"
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		t.Fatalf("Failed to create output directory: %v", err)
	}

	original := generateBassMelody(duration, sampleRate, freq1, freq2)

	// Parameters for Shimmer
	delayTime := 0.3            // seconds
	mix := 0.5                  // 50% shimmer mix
	pitchShiftSemitones := 12.0 // One octave up
	feedback := 0.3             // 30% feedback

	// Parameters for Bitcrusher
	bitDepth := 4              // Lower bit depth for noticeable quantization noise
	sampleRateReduction := 100 // Higher sample rate reduction for heavy bitcrushing

	// Apply ShimmerBitcrusher
	shimmerBitcrushed := ShimmerBitcrusher(
		append([]float64(nil), original...),
		sampleRate,
		delayTime,
		mix,
		pitchShiftSemitones,
		bitDepth,
		sampleRateReduction,
		feedback,
	)
	shimmerBitcrusherPath := filepath.Join(outputDir, "shimmer_bitcrusher.wav")
	if err := writeWav(shimmerBitcrusherPath, shimmerBitcrushed, sampleRate, 1); err != nil {
		t.Fatalf("Failed to write shimmer_bitcrusher.wav: %v", err)
	}
	t.Logf("Generated %s", shimmerBitcrusherPath)
}

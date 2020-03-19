import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;

import 'package:fft/fft.dart';

/// Returns signal data as a list from a recording at [filepath].
Future<List<num>> getSignalFromFile(String filepath) async {
  // Number reserved for the metadata coming from the audio files.
  final waveMetadataOffset = 44;
  final sampleRate = 16000;
  final file = File(filepath).readAsBytesSync();
  final soundBuffer = file.buffer.asInt16List(waveMetadataOffset);

  // Padding to ensure exactly one second of sound for all
  // audio files.
  return List<num>(sampleRate)
      .asMap()
      .map(
        (index, value) => MapEntry(
          index,
          index < soundBuffer.length ? soundBuffer[index] : 0,
        ),
      )
      .values
      .toList();
}

/// Converting the [signal] to a tensor representation.
///
/// Takes the audio input signal and transfers it to a
/// 2D representation by applying a sliding Hann window
/// and calculating the Fourier transform.
List<double> signalToSpectrogram(List<num> signal) {
  // Specifies the half of a window length, as the first
  // half of the data in each FFT is duplicated from the last
  // half of the previous FFT window.
  final overlap = 64;
  final winSize = overlap * 2;
  final freqLen = winSize ~/ 2 + 1;
  final epsilon = math.pow(10, -8);
  final loops = signal.length ~/ overlap;
  final window = Window(WindowType.HANN);
  List<double> results = [];

  // Applies a sliding Hann window and calculates the
  // Fourier transform for a number of frequencies in the
  // spectrum of 0 to 16000 Hz.
  for (int i = 1; i < loops; i++) {
    int start = i * overlap - overlap;
    int end = start + winSize;
    List<num> chunk = signal.sublist(start, end);
    List<double> fft =
        FFT().Transform(window.apply(chunk)).sublist(0, freqLen).map((e) {
      double val = math.log(e.modulus + epsilon);
      return double.parse("$val");
    }).toList();

    results.addAll(fft);
  }
  return results;
}

/// Converts the incoming [spectrogram] to type `Int8List`.
Int8List spectrogramToTensor(List<double> spectrogram) {
  return Float32List.fromList(spectrogram).buffer.asInt8List();
}

class Prediction {
  final String className;
  final double confidence;

  Prediction(this.className, this.confidence);
}

/// Puts the [confidences] and [classes] into a list of `Prediction`.
///
/// Maps the confidences to the corresponding classes by using the
/// index in a list of `Prediction`.
List<Prediction> processPredictions(
  Float32List confidences,
  List<String> classes,
) {
  return List<double>.from(confidences)
      .asMap()
      .map(
        (index, confidence) => MapEntry(
          index,
          Prediction(
            classes[index],
            confidence,
          ),
        ),
      )
      .values
      .toList()
        ..sort((a, b) => b.confidence.compareTo(a.confidence))
        ..toList();
}

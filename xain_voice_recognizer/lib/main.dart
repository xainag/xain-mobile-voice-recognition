import 'dart:io';
import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_audio_recorder/flutter_audio_recorder.dart';
import 'package:audioplayers/audioplayers.dart';

import './tflite/tflite.dart' as tfl;
import './tflite/src/tensor.dart';
import './widgets/word_tile.dart';
import './widgets/info_text.dart';
import './utils.dart';
import './classes.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Xain Voice',
      theme: ThemeData(
        primarySwatch: Colors.blueGrey,
      ),
      home: MyHomePage(title: 'Xain Voice'),
    );
  }
}

enum AppState {
  IsInitializing,
  IsError,
  IsReady,
  IsRecording,
  IsPlaying,
  IsPredicting
}

/// The duration for a single recording.
const RECORDING_DURATION = Duration(seconds: 1);

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);

  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  AppState _appState = AppState.IsInitializing;
  tfl.Interpreter _interpreter;
  List<Prediction> _predictions = [];
  FlutterAudioRecorder _recorder;
  Recording _recording;
  String _feedbackMessage;

  @override
  void initState() {
    super.initState();

    Future.microtask(() async {
      try {
        await _initializeInterpreter();

        /// True if the recorder was initialized successfully.
        var hasPermissions = await _checkPermissionsForRecorder();

        if (hasPermissions) {
          setState(() {
            _appState = AppState.IsReady;
          });
        } else {
          setState(() {
            _appState = AppState.IsError;
            _feedbackMessage = 'You need to give the app proper permissions';
          });
        }
      } catch (e) {
        print(e);

        setState(() {
          _appState = AppState.IsError;
          _feedbackMessage = e.message;
        });
      }
    });
  }

  /// Initializes the interpreter by loading the voice model and allocating tensors.
  Future<void> _initializeInterpreter() async {
    String appDirectory = (await getApplicationDocumentsDirectory()).path;
    String srcPath = "assets/digitsnet.tflite";
    String destPath = "$appDirectory/model.tflite";

    /// Read the model as bytes and write it to a file in a location
    /// which can be accessed by TFLite native code.
    ByteData modelData = await rootBundle.load(srcPath);
    await File(destPath).writeAsBytes(modelData.buffer.asUint8List());

    /// Initialise the interpreter
    _interpreter = tfl.Interpreter.fromFile(destPath);
    _interpreter.allocateTensors();
  }

  /// Returns `true` if the user allowed for recording permissions.
  Future<bool> _checkPermissionsForRecorder() async {
    // Asks for proper permissions from the user
    return await FlutterAudioRecorder.hasPermissions;
  }

  /// Initializes the recorder and deletes the last recording.
  Future<void> _initializeRecorder() async {
    String dirPath = (await getApplicationDocumentsDirectory()).path;
    String filename = 'tempFile.wav';
    String filePath = "$dirPath/$filename";
    File file = File(filePath);

    /// If the file exists, clean it up first
    /// because the FlutterAudioRecorder will
    /// throw an error if the file exists
    if (file.existsSync()) {
      file.deleteSync();
    }

    _recorder = FlutterAudioRecorder(
      filePath,
      audioFormat: AudioFormat.WAV,
    );

    await _recorder.initialized;
  }

  /// Spins the recorder up and clears the previous predictions.
  Future<void> _startRecording() async {
    await _initializeRecorder();
    await _recorder.start();

    /// The current status of the recording.
    Recording recordingState = await _recorder.current();

    setState(() {
      _appState = AppState.IsRecording;
      _recording = recordingState;

      // We clear results so we can show the recording status.
      _predictions.clear();
    });
  }

  /// Sets the recorder into a finalizing state.
  Future<void> _stopRecording() async {
    Recording recordingState = await _recorder.stop();

    setState(() {
      _recording = recordingState;
    });
  }

  /// Retrieves the confidences of the word classes for the latest recording.
  ///
  /// Reads the recording signal from the file, converts it to a spectrogram,
  /// tensor data afterwards and gets the confidences for the input.
  Future<void> _performPrediction() async {
    setState(() {
      _appState = AppState.IsPredicting;
    });

    try {
      // Retrieves the tensor data for the last recording.
      List<num> signalData = await getSignalFromFile(_recording?.path ?? '');
      List<double> spectrogram = signalToSpectrogram(signalData);
      Int8List inputData = spectrogramToTensor(spectrogram);

      // The data is passed into the interpreter, which runs inference for loaded graph.
      List<Tensor> inputTensors = _interpreter.getInputTensors();
      inputTensors[0].data = inputData;
      _interpreter.invoke();

      // Get results and parse them into relations of confidences to classes.
      List<Tensor> outputTensors = _interpreter.getOutputTensors();
      Float32List outputData = outputTensors[0].data.buffer.asFloat32List();
      List<Prediction> predictions = processPredictions(outputData, classes);

      /// The prediction sublist in order to just show the first three top confidences.
      predictions = predictions.sublist(0, 3);

      setState(() {
        _appState = AppState.IsReady;
        _predictions = predictions;
      });
    } catch (e) {
      print(e);

      setState(() {
        _appState = AppState.IsError;
        _feedbackMessage = e.message;
      });
    }
  }

  // Records the user's voice and computes the confidences for it.
  Future<void> _recordAudio() async {
    await _startRecording();
    await Future.delayed(RECORDING_DURATION);
    await _stopRecording();
    await _performPrediction();
  }

  /// Plays the last recording.
  Future<void> _playAudio() async {
    setState(() {
      _appState = AppState.IsPlaying;
    });

    AudioPlayer currentPlayer = AudioPlayer();
    await currentPlayer.play(_recording.path, isLocal: true);

    currentPlayer.onPlayerCompletion.listen((event) async {
      await currentPlayer.dispose();

      setState(() {
        _appState = AppState.IsReady;
      });
    });
  }

  List<Widget> _statusText(AppState state) {
    switch (state) {
      case AppState.IsError:
        return [
          InfoText('ERROR:'),
          InfoText(_feedbackMessage ?? 'Something went wrong'),
        ];
      case AppState.IsPredicting:
        return [
          InfoText('Calculating probability...'),
        ];
      case AppState.IsRecording:
        return [
          InfoText('Recording...'),
        ];
      case AppState.IsPlaying:
        return [
          InfoText('Playing...'),
        ];
      case AppState.IsInitializing:
        return [
          InfoText('Model is loading'),
        ];
      case AppState.IsReady:
      default:
        return [
          InfoText('You can record your word.'),
          InfoText('Recording takes 1 second.'),
        ];
    }
  }

  IconData _recordIcon(AppState state) {
    switch (state) {
      case AppState.IsReady:
        return Icons.mic;
      case AppState.IsRecording:
      case AppState.IsPredicting:
        return Icons.mic_none;
      default:
        return Icons.mic_off;
    }
  }

  IconData _playIcon(AppState state) {
    switch (state) {
      case AppState.IsPlaying:
        return Icons.pause;
      default:
        return Icons.play_arrow;
    }
  }

  @override
  Widget build(BuildContext context) {
    bool _canRecord = _appState == AppState.IsReady;
    bool _canPlay = _canRecord && (_recording?.path?.endsWith('.wav') ?? false);

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        centerTitle: true,
      ),
      backgroundColor: Colors.blueGrey[50],
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          Container(
            alignment: Alignment.center,
            padding: const EdgeInsets.all(24),
            child: Column(
              children: <Widget>[
                InfoText(
                    'Press "Record" and say one of the ${classes.length} words below.'),
                InfoText('Xain Voice will predict which word it was.'),
              ],
            ),
          ),
          Container(
            padding: EdgeInsets.symmetric(
              horizontal: 24,
            ),
            alignment: Alignment.center,
            child: Wrap(
              alignment: WrapAlignment.center,
              spacing: 8,
              runSpacing: 8,
              children: classes.map<Widget>(
                (String word) {
                  bool isSelected = _predictions.isNotEmpty &&
                          word == _predictions.first.className ??
                      false;
                  return WordTile(
                    word,
                    isSelected: isSelected,
                  );
                },
              ).toList(),
            ),
          ),
          Container(
            padding: EdgeInsets.symmetric(
              vertical: 24,
            ),
            alignment: Alignment.center,
            height: 102,
            child: Column(
              children: (_appState == AppState.IsReady ||
                          _appState == AppState.IsPlaying) &&
                      _predictions.isNotEmpty
                  ? _predictions.map((prediction) {
                      String className = prediction.className;
                      String confidence =
                          (prediction.confidence * 100).toStringAsFixed(2) +
                              "%";
                      return InfoText("$className = $confidence");
                    }).toList()
                  : _statusText(_appState),
            ),
          ),
          Column(
            children: <Widget>[
              Padding(
                padding: EdgeInsets.all(12),
                child: FloatingActionButton(
                  /// Calls _recordAudio function if the record button
                  /// is pressed and the app is in ready state.
                  onPressed: _canRecord ? _recordAudio : null,
                  tooltip: 'Record voice',
                  child: Icon(
                    _recordIcon(_appState),
                    color: _canRecord ? Colors.black54 : Colors.black26,
                  ),
                  backgroundColor:
                      _canRecord ? Colors.amber : Colors.blueGrey[100],
                ),
              ),
              Padding(
                padding: EdgeInsets.all(12),
                child: FloatingActionButton(
                  onPressed: _canPlay ? _playAudio : null,
                  mini: true,
                  tooltip: 'Play voice',
                  child: Icon(
                    _playIcon(_appState),
                    color: _canPlay ? Colors.black54 : Colors.black26,
                  ),
                  backgroundColor:
                      _canPlay ? Colors.amber : Colors.blueGrey[100],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    // Friendly deletion of interpreter instance
    // and shutting down of audio player.
    _interpreter.delete();

    super.dispose();
  }
}

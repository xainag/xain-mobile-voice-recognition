<h1 align="center">
  <br>
  XAIN Mobile Voice Recognition
  <br>
</h1>

<p align="center">
  <a href="#how-does-it-work">How does the application work</a> •
  <a href="#getting-started">Getting started</a> •
  <a href="#starting-application">Starting the application</a> •
  <a href="#license">License</a>
</p>

<p align="center">
The XAIN Mobile Voice Recognizer is a mobile speech recognition application written in Dart. Depending on the user's recorded voice it predicts which word out of 15 was recorded. Therefore, TFLite methods exposed by the binary are called with the help of the bindings provided by <a href="https://github.com/dart-lang/tflite_native">tflite_native</a>.
For further information like how the model was built or why we decided to follow a cross-platform with Flutter, refer to our "How we did it." article.
</p>

<p align="center">
  <img src="https://github.com/xainag/xain-mobile-voice-recognition/blob/AP-183_create_documentation/demo/xain_voice_demo.gif" width="275" height="550" />
</p>


## How does the application work

XAIN's mobile voice recognizer is able to predict simple recorded words depending on the user's voice by using a converted voice classification model under the hood. Once the application is started, the user can record one of the 15 words and immediately gets the prediction, which word was recorded. Hence, the word with the highest confidence is highlighted and the top three predictions are displayed. By pressing the play button the user can hear the previously recorded voice again.

## Getting started

### Installing Flutter

First you need Flutter installed on your machine since it is used as toolkit for building our mobile application. Therefore, follow [here](https://flutter.dev/docs/get-started/install/macos) to install Flutter and necessary mobile environments. Afterwards you can run this command to check if your environment is properly configured: 

```shell
flutter doctor
```

It may happen that you get `! No devices available` warning as part of this commands' output. Don't worry, it only means that you don't have an emulator running or a physical device connected to your machine currently.


### Installing from source

Clone this repository:

```shell
git clone https://github.com/xainag/xain-mobile-voice-recognition.git
```

## Starting the application

In order to start the application you need to first launch an emulator or connect your physical Android device to the machine. Regarding the emulator it can be either Android Emulator or iOS Simulator.

Then you can run the project using the command line:

```shell
cd xain_voice_recognizer
flutter run
```

This should build your application and run it on the connected emulator.

## License

Apache License 2.0

## Further info

This project is a Flutter application. For help getting started with it, view the [online documentation](https://flutter.dev/docs), which offers tutorials, samples, guidance on mobile development, and a full API reference.
# XAIN Mobile Voice Recognition

The XAIN Mobile Voice Recognizer is a mobile speech recognition application written in Dart. Depending on the user's recorded voice it predicts which word out of 15 was recorded. Therefore, TFLite methods exposed by the binary are called with the help of the bindings provided by [tflite_native](https://github.com/dart-lang/tflite_native). For further information like how the model was built or why we decided to follow a cross-platform with Flutter, refer to our "How we did it." article.

<img src="https://github.com/xainag/xain-mobile-voice-recognition/blob/master/demo/xain_voice_demo.gif" width="40" height="80" />

## How does the application work

XAIN's mobile voice recognizer is able to predict simple recorded words depending on the user's voice by using  Once the application is started, the user can record one of the 15 words and immediately gets the prediction, which word was recorded. Hence, the word with the highest confidence is highlighted and the top three predictions are displayed. By pressing the play button the user can hear the previously recorded voice again.

## Getting started

### Installing Flutter

First you need Flutter installed on your machine since it is used as toolkit for building our mobile application. Install Flutter and necessary mobile dev environments https://flutter.dev/docs/get-started/install/macos You can run this command to check if your development environment is properly configured: 

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

In order to start the application you need to first launch an emulator or connect your physical Android device to the machine. Regarding the emulator can be either Android Emulator or iOS Simulator. You can launch it directly from your VS Code if you've installed the Flutter plugin.

Then you can run the project using the command line:

```shell
flutter run
```

This should build your application and run it on the connected emulator.

## License

## Further info

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://flutter.dev/docs/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://flutter.dev/docs/cookbook)

For help getting started with Flutter, view our [online documentation](https://flutter.dev/docs), which offers tutorials, samples, guidance on mobile development, and a full API reference.
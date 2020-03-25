<h1 align="center">
  <br>
  XAIN Mobile Voice Recognition
  <br>
</h1>

<p align="center">
  <a href="#how-does-the-application-work">How does the application work</a> •
  <a href="#getting-started">Getting started</a> •
  <a href="#starting-the-application">Starting the application</a> •
  <a href="#license">License</a>
</p>

<p align="left">
The XAIN Mobile Voice Recognizer is a mobile speech recognition application written in Dart. It uses Flutter as mobile UI toolkit and TFLite as mobile machine learning framework for doing inference. Depending on the user's recorded voice it predicts which word out of 15 was recorded. Therefore, methods exposed by the TFLite binary are called with the help of the bindings provided by <a href="https://github.com/dart-lang/tflite_native">tflite_native</a>. One of the reasons for adding this example is to show other developers how to run inference on mobiles with their own custom model.

The model has been trained using <a href="https://github.com/xainag/xain-mobile-voice-recognition/tree/master/python_files/tfl_converter.py">this python file</a>. Feel free to change the model and experiment with it.

For further information like how the model was built or why we decided to follow a cross-platform with Flutter, refer to our "How we did it." article.
</p>

<p align="center">
  <img src="https://github.com/xainag/xain-mobile-voice-recognition/blob/AP-183_create_documentation/demo/xain_voice_demo.gif" width="275" height="550" />
</p>


## How does the application work

XAIN's mobile voice recognizer is able to predict simple recorded words depending on the user's voice by using a converted voice classification model under the hood. Once the application is started, the user can record one of the 15 words and immediately gets the prediction, which word was recorded. Hence, the word with the highest confidence is highlighted and the top three predictions are displayed. By pressing the play button the user can hear the previously recorded voice again.

## Getting started

### Installing Flutter

First you need Flutter installed on your machine since it is used as toolkit for building our mobile application. Therefore, follow [here](https://flutter.dev/docs/get-started/install) to install Flutter and necessary mobile environments. The application is able to run on both mobile operating systems, Android and iOS. After finishing the installation process you can run this command to check if your environment is properly configured: 

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

In order to start the application you need to first connect your physical Android or iOS device to the machine or launch an emulator. The type of virtual device depends on your operating system and can be either Android Emulator, iOS Simulator or both. For further information about setting up virtual devices refer to [Flutter installation](https://flutter.dev/docs/get-started/install) guidelines.

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
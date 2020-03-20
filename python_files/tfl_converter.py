"""Fit a speech recognizer model and covert it to TFLite."""

import pathlib
import pandas as pd
import numpy as np
import scipy.io.wavfile
import sklearn.preprocessing

from tqdm import tqdm

import tensorflow as tf

# from tensorflow.keras import Sequential
# from tensorflow.keras.callbacks import History
# from tensorflow.keras.layers import (
#     Dense,
#     Dropout,
#     BatchNormalization,
#     Conv1D,
#     LSTM,
#     Input,
#     Activation,
#     MaxPool1D,
#     Reshape,
# )
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.activations import relu
# from tensorflow.keras.models import Model
# from tensorflow import Tensor

from typing import Tuple, List, Dict

import structlog

LOG = structlog.get_logger()

import pdb



class SpeechModelTFLiteConverter:

    SAMPLE_RATE = 16000
    WINDOWS_SIZE = 128
    EPSILON = 1e-7
    N_WORDS = 15

    def __init__(self):
        data_dir = pathlib.Path(__file__).parent / "data"
        self.train_file_names_path = data_dir / "train_files.csv"
        self.test_file_names_path = data_dir / "test_files.csv"
        self.audio_dir = data_dir / "audio"
        self.overlap = self.WINDOWS_SIZE // 2
        self.time_samples = self.SAMPLE_RATE // (self.WINDOWS_SIZE - self.overlap)
        self.tflite_file_path = data_dir / "speech_model.tflite"

    def calculate_log_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Calculates log_spectrogram (frequencies over time plot) of the audio signal.

        Args:
            audio (np.ndarray): audio signal

        Returns:
            log_spectrogram: logarithm of the spectrogram
        """

        number_of_frequencies = int(self.WINDOWS_SIZE / 2) + 1
        window = np.hanning(self.WINDOWS_SIZE)
        log_spectrogram = np.empty((self.time_samples - 1, number_of_frequencies))

        for i in range(1, self.time_samples):
            # Loops over time windows and calculates the FFT of the function * Hann Window.
            # After that we take only the first half of the frequencies as FFT is symmetrical
            # and calculate the log of shifted function
            start = int((i - 1) * self.WINDOWS_SIZE / 2)
            end = int((i + 1) * self.WINDOWS_SIZE / 2)
            func = audio[start:end]
            transformed_signal = np.absolute(np.fft.fft(func * window)[:number_of_frequencies])
            log_spectrogram[i - 1, :] = np.log(self.EPSILON + transformed_signal)

        return log_spectrogram

    def pad_audio(self, samples: np.ndarray) -> np.ndarray:
        """Pads the audio file in case the length is not set to 1 sec.

        Args:
            samples (np.ndarray): Audio signal

        Returns:
            padded_samples (np.ndarray): padded signal
        """
        if len(samples) >= self.SAMPLE_RATE:
            padded_samples = samples
        else:
            padded_samples = np.pad(
                samples,
                pad_width=(self.SAMPLE_RATE - len(samples), 0),
                mode="constant",
                constant_values=(0, 0),
            )
        return padded_samples

    def read_vaw_files(self, files_sample: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Reads files from the dataframe, pads them and transforms the log_spectrograms
            to ndarrays.

        Args:
            files_sample (pd.DataFrame): DataFrame containing the file names of the audio files,
                mapped to the spoken word in the audio.

        Returns:
            x_array (np.ndarray): Array of containing log_spectrograms for each file.
            y_array (List[str]): List of the corresponding classes (spoken words) for each file.
        """

        resized_overlap = self.overlap + 1
        n_samples = len(files_sample)
        x_array = np.empty((n_samples, self.time_samples - 1, resized_overlap))
        y_array = []

        LOG.info("processing audio files")
        for i, row in tqdm(files_sample.iterrows(), total=n_samples):
            file_path = self.audio_dir / row["class"] / row["filename"]
            sample_rate, samples = scipy.io.wavfile.read(file_path)
            padded_samples = self.pad_audio(samples)
            spectrogram = self.calculate_log_spectrogram(padded_samples)

            # add samples and truncate them when too long
            x_array[i, :, :] = spectrogram[:, :resized_overlap]
            y_array.append(row["class"])

        return x_array, y_array

    def encode_ys(self, y_train: List[str], y_test: List[str], all_words: pd.DataFrame):
        """Encode Ys.

        Args:
            y_train (List[str]): List of classes for the training set.
            y_test (List[str]): List of classes for the test set.
            all_words (pd.DataFrame): DataFrame containing all the words.

        Returns:
            y_train_transformed (np.ndarray): Array of encoded Ys for the training set.
            y_test_transformed (np.ndarray): Array of encoded Ys for the test set.
        """

        encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoder.fit(all_words)

        y_train_transformed = encoder.transform(pd.DataFrame(y_train))
        y_test_transformed =  encoder.transform(pd.DataFrame(y_test))

        return y_train_transformed, y_test_transformed

    @staticmethod
    def reduce_conv(tensor: tf.Tensor, num_filters: int, k: int, strides: int = 2, padding: str = "valid"):
        """ Building block of the neurat network. It includes the Conv1D layer + normalization+relu  + max pooling

        Args:
            tensor (tf.Tensor): Tensor
            num_filters (int): Number of filters in the convolutional layer
            k (int): The number of positions by which the filter is moved right at each step.
            strides (int):  Maxx pooling strides
            padding (str):  Type of padding used

        Returns:
            tensor (tf.Tensor): Tensor after the Conv1D layer.
        """
        tensor = tf.keras.layers.Conv1D(
            num_filters,
            k,
            padding=padding,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.00002),
        )(tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(tensor)
        tensor = tf.keras.layers.MaxPool1D(pool_size=3, strides=strides, padding=padding)(tensor)

        return tensor

    @staticmethod
    def context_conv(tensor: tf.Tensor, num_filters: int, k: int, dilation_rate: int = 1, padding: str = "valid"):
        """ Building block of the neurat network. It includes the Conv1D layer + normalization+relu activation function

        Args:
            tensor (tf.Tensor): Tensor.
            num_filters: Number of filters in the convolutional layer.
            k (int): The number of positions by which the filter is moved right at each step.
            dilation_rate (int):  Number of skipped cells between filters.
            padding (str):  Type of padding used.

        Returns:
            tensor (tf.Tensor): Tensor after the Conv1D layer.
        """

        tensor = tf.keras.layers.Conv1D(
            num_filters,
            k,
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_regularizer=tf.keras.regularizers.l2(0.00002),
            use_bias=False,
        )(tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(tensor)

        return tensor

    def conv_1d_time_stacked_model(self, input_size: Tuple[int, int]):
        """ Creates a 1D model for temporal data.

        Args:
            input_size (Tuple[int, int]): Size of the input vector.

        Returns:
            model (Model): Compiled Keras model.
        """

        input_layer = tf.keras.layers.Input(shape=input_size)
        tensor = input_layer

        tensor = self.context_conv(tensor, num_filters=16, k=1)
        tensor = self.reduce_conv(tensor, num_filters=32, k=3)
        tensor = self.context_conv(tensor, num_filters=32, k=3)

        tensor = self.reduce_conv(tensor, num_filters=64, k=3)
        tensor = tf.keras.layers.Dropout(0.1)(tensor)
        tensor = self.context_conv(tensor, num_filters=64, k=3)

        tensor = self.reduce_conv(tensor, num_filters=128, k=3)
        tensor = tf.keras.layers.Dropout(0.1)(tensor)
        tensor = self.context_conv(tensor, num_filters=128, k=3)

        tensor = self.reduce_conv(tensor, num_filters=249, k=3)
        tensor = self.context_conv(tensor, num_filters=249, k=3)

        tensor = tf.keras.layers.Dropout(0.1)(tensor)
        tensor = tf.keras.layers.Conv1D(self.N_WORDS, 9, activation="softmax")(tensor)
        tensor = tf.keras.layers.Reshape([-1])(tensor)

        model = tf.keras.models.Model(input_layer, tensor, name="conv_1d_time_stacked")
        model.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )
        return model

    def prepare_data(self):
        train_sample_files = pd.read_csv(self.train_file_names_path, index_col=0)
        test_sample_files = pd.read_csv(self.test_file_names_path, index_col=0)
        x_train, y_train, = self.read_vaw_files(train_sample_files)
        x_test, y_test = self.read_vaw_files(test_sample_files)

        words_1_15 = train_sample_files["class"].unique().tolist()
        all_words = pd.DataFrame(words_1_15)
        y_train_encoded, y_test_encoded = self.encode_ys(y_train, y_test, all_words)

        data = {
            "x_train": x_train,
            "y_train": y_train_encoded,
            "x_test": x_test,
            "y_test": y_test_encoded
        }
        return data

    def fit_model(self, data: Dict[str, np.ndarray]):

        spectrogram_shape = data["x_train"].shape[1:]
        recognizer = self.conv_1d_time_stacked_model(spectrogram_shape)

        recognizer.fit(
            data["x_train"],
            data["y_train"],
            batch_size=128,
            epochs=30,
            verbose=1,
            validation_data=(data["x_test"], data["y_test"]),
            shuffle=True,
        )

        return recognizer

    def convert(self):
        data = self.prepare_data()
        recognizer = self.fit_model(data)

        converter = tf.lite.TFLiteConverter.from_keras_model(recognizer)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(str(self.tflite_file_path), "wb") as f:
            f.write(tflite_model)
        LOG.info("TFLite model written to: {}".format(self.tflite_file_path))


def main():
    speech_model_converter = SpeechModelTFLiteConverter()
    speech_model_converter.convert()


if __name__ == "__main__":
    main()





# train_sample = pd.read_csv(
#     "xain_benchmark/testcases/keras/speech_recognition/split_data/words/words_1-15.csv"
# )
# test_sample = pd.read_csv(
#     "xain_benchmark/testcases/keras/speech_recognition/split_data/test_1-15.csv"
# )
# words_1_15 = test_sample["class"].unique().tolist()
# train_sample = train_sample.loc[train_sample["class"].isin(words_1_15)]
# all_words = pd.DataFrame(words_1_15)
#
# x_train, y_train, = utils.read_vaw_files(train_sample)
# x_test, y_test = utils.read_vaw_files(test_sample)
#
# enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
# enc.fit(all_words)
#
# y_train_transformed = enc.transform(pd.DataFrame(y_train))
# y_test_transformed =  enc.transform(pd.DataFrame(y_test))
#
# input = (x_train.shape[1], x_train.shape[2])
# recognizer = utils.conv_1d_time_stacked_model(input, num_classes=15)
#
# history = recognizer.fit(
#             x_train,
#             y_train_transformed,
#             batch_size=128,
#             epochs=30,
#             verbose=1,
#             validation_data=(x_test, y_test_transformed),
#             shuffle=True,
#         )
# converter = tf.lite.TFLiteConverter.from_keras_model(recognizer)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()
# with open("{}/speech_model.tflite".format('xain_benchmark'),"wb") as f:
#     f.write(tflite_model)


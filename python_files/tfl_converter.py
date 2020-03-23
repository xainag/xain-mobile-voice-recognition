"""Fit a speech recognizer model and covert it to TFLite."""

import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io.wavfile
import sklearn.preprocessing
import structlog
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dropout,
    Input,
    MaxPool1D,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tqdm import tqdm

LOG = structlog.get_logger()


class SpeechModelTFLiteConverter:
    """Prepare the audio data, initialize and fit the model, and convert the Keras model to TFLite.

    The class expects the audio data to be available locally under:
    'xain-mobile-voice-recognition/python_files/data/audio/'.

    The data can be found at: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data

    It will preprocess the data so that the audio files are read to a buffer and converted
    to spectrograms. The spectrograms are then fed to the a speech recognition custom Keras model,
    the model is trained, and converted to a TFLite file, which is saved locally.
    """

    SAMPLE_RATE = 16000
    WINDOWS_SIZE = 128
    EPSILON = 1e-7
    N_WORDS = 15
    EPOCHS = 30

    def __init__(self) -> None:
        data_dir = pathlib.Path(__file__).parent / "data"
        self.train_file_names_path = data_dir / "train_files.csv"
        self.test_file_names_path = data_dir / "test_files.csv"
        self.audio_dir = data_dir / "audio"
        self.overlap = self.WINDOWS_SIZE // 2
        self.time_samples = self.SAMPLE_RATE // (self.WINDOWS_SIZE - self.overlap)
        self.tflite_file_path = data_dir / f"speech_model_{self.EPOCHS}_epochs.tflite"

    def calculate_log_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Calculates log_spectrogram (frequencies over time plot) of the audio signal.

        Args:
            audio (np.ndarray): audio signal.

        Returns:
            log_spectrogram (np.ndarray): logarithm of the spectrogram.
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
            transformed_signal = np.absolute(
                np.fft.fft(func * window)[:number_of_frequencies]
            )
            log_spectrogram[i - 1, :] = np.log(self.EPSILON + transformed_signal)

        return log_spectrogram

    def pad_audio(self, samples: np.ndarray) -> np.ndarray:
        """Pads the audio file in case the length is not set to 1 sec.

        Args:
            samples (np.ndarray): Audio signal.

        Returns:
            padded_samples (np.ndarray): Padded signal.
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

    def process_vaw_files(
            self, files_sample: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Reads files from the dataframe, pads them and transforms the log_spectrograms
        to ndarrays.

        Args:
            files_sample (pd.DataFrame): DataFrame containing the file names of the audio files,
                mapped to the spoken word in the audio.

        Returns:
            x_array (np.ndarray): Array of containing log_spectrograms for each file.
            y_list (List[str]): List of the corresponding classes (spoken words) for each file.
        """

        resized_overlap = self.overlap + 1
        n_samples = len(files_sample)
        x_array = np.empty((n_samples, self.time_samples - 1, resized_overlap))
        y_list = []

        LOG.info("processing audio files")
        for i, row in tqdm(files_sample.iterrows(), total=n_samples):
            file_path = self.audio_dir / row["class"] / row["filename"]
            _, samples = scipy.io.wavfile.read(file_path)
            padded_samples = self.pad_audio(samples)
            spectrogram = self.calculate_log_spectrogram(padded_samples)

            # add samples and truncate them when too long
            x_array[i, :, :] = spectrogram[:, :resized_overlap]
            y_list.append(row["class"])

        return x_array, y_list

    @staticmethod
    def encode_ys(
            y_train: List[str], y_test: List[str], all_words: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode Ys.

        Args:
            y_train (List[str]): List of classes for the training set.
            y_test (List[str]): List of classes for the test set.
            all_words (pd.DataFrame): DataFrame containing all the words.

        Returns:
            y_train_transformed (np.ndarray): Array of encoded Ys for the training set.
            y_test_transformed (np.ndarray): Array of encoded Ys for the test set.
        """

        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown="ignore", sparse=False
        )
        encoder.fit(all_words)

        y_train_transformed = encoder.transform(pd.DataFrame(y_train))
        y_test_transformed = encoder.transform(pd.DataFrame(y_test))

        return y_train_transformed, y_test_transformed

    @staticmethod
    def reduce_conv(tensor: Tensor, num_filters: int, k: int) -> Tensor:
        """ Building block of the neurat network. It includes the Conv1D layer + normalization +
        relu + max pooling.

        Args:
            tensor (Tensor): Tensor.
            num_filters (int): Number of filters in the convolutional layer.
            k (int): The number of positions by which the filter is moved right at each step.

        Returns:
            tensor (Tensor): Tensor after the Conv1D layer.
        """
        tensor = Conv1D(
            num_filters,
            k,
            padding="valid",
            use_bias=False,
            kernel_regularizer=l2(0.00002),
        )(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Activation(relu)(tensor)
        tensor = MaxPool1D(pool_size=3, strides=2, padding="valid")(tensor)

        return tensor

    @staticmethod
    def context_conv(tensor: Tensor, num_filters: int, k: int) -> Tensor:
        """ Building block of the neurat network. It includes the Conv1D layer + normalization +
         relu activation function

        Args:
            tensor (tf.Tensor): Tensor.
            num_filters: Number of filters in the convolutional layer.
            k (int): The number of positions by which the filter is moved right at each step.

        Returns:
            tensor (Tensor): Tensor after the Conv1D layer.
        """

        tensor = Conv1D(
            num_filters,
            k,
            padding="valid",
            dilation_rate=1,
            kernel_regularizer=l2(0.00002),
            use_bias=False,
        )(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Activation(relu)(tensor)

        return tensor

    def conv_1d_time_stacked_model(self, input_size: Tuple[int, int]) -> Model:
        """ Creates a 1D model for temporal data.

        Args:
            input_size (Tuple[int, int]): Size of the input vector.

        Returns:
            model (Model): Compiled Keras model.
        """

        input_layer = Input(shape=input_size)
        tensor = input_layer

        tensor = self.context_conv(tensor, num_filters=16, k=1)
        tensor = self.reduce_conv(tensor, num_filters=32, k=3)
        tensor = self.context_conv(tensor, num_filters=32, k=3)

        tensor = self.reduce_conv(tensor, num_filters=64, k=3)
        tensor = Dropout(0.1)(tensor)
        tensor = self.context_conv(tensor, num_filters=64, k=3)

        tensor = self.reduce_conv(tensor, num_filters=128, k=3)
        tensor = Dropout(0.1)(tensor)
        tensor = self.context_conv(tensor, num_filters=128, k=3)

        tensor = self.reduce_conv(tensor, num_filters=249, k=3)
        tensor = self.context_conv(tensor, num_filters=249, k=3)

        tensor = Dropout(0.1)(tensor)
        tensor = Conv1D(self.N_WORDS, 9, activation="softmax")(tensor)
        tensor = Reshape([-1])(tensor)

        model = Model(input_layer, tensor, name="conv_1d_time_stacked")
        model.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )
        return model

    def prepare_data(self) -> Dict[str, np.ndarray]:
        """Prepare data by reading the audio files, processing them and encoding the Y labels.

        Returns:
            data (Dict[str, np.ndarray]): Dictionary of data split between training and test, and
                between X and y.
        """

        train_sample_files = pd.read_csv(self.train_file_names_path, index_col=0)
        test_sample_files = pd.read_csv(self.test_file_names_path, index_col=0)
        x_train, y_train, = self.process_vaw_files(train_sample_files)
        x_test, y_test = self.process_vaw_files(test_sample_files)

        words_1_15 = train_sample_files["class"].unique().tolist()
        all_words = pd.DataFrame(words_1_15)
        y_train_encoded, y_test_encoded = self.encode_ys(y_train, y_test, all_words)

        data = {
            "x_train": x_train,
            "y_train": y_train_encoded,
            "x_test": x_test,
            "y_test": y_test_encoded,
        }
        return data

    def fit_model(self, data: Dict[str, np.ndarray]) -> Model:
        """Initialise the model and fit it with the training data.

        Args:
            data (Dict[str, np.ndarray]): Dictionary containing training and test data,
                for X and y.

        Returns:
            recognizer (Model): Fitted Keras model.
        """

        spectrogram_shape = data["x_train"].shape[1:]
        recognizer = self.conv_1d_time_stacked_model(spectrogram_shape)

        recognizer.fit(
            data["x_train"],
            data["y_train"],
            batch_size=128,
            epochs=self.EPOCHS,
            verbose=1,
            validation_data=(data["x_test"], data["y_test"]),
            shuffle=True,
        )

        return recognizer

    def convert(self) -> None:
        """Main function to prepare the data, initialize and fit the model,
        and convert the Keras model to TFLite, saving the file locally.
        """

        data = self.prepare_data()
        recognizer = self.fit_model(data)

        converter = tf.lite.TFLiteConverter.from_keras_model(recognizer)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(str(self.tflite_file_path), "wb") as file:
            file.write(tflite_model)
        LOG.info("TFLite model written to: {}".format(self.tflite_file_path))


def main():
    """Main function to run the script."""

    speech_model_converter = SpeechModelTFLiteConverter()
    speech_model_converter.convert()


if __name__ == "__main__":
    main()

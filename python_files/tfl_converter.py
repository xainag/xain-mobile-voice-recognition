# import pandas as pd
# import python_files.utils
# import tensorflow as tf
#
# from sklearn.preprocessing import OneHotEncoder

import pathlib
import pandas as pd
import numpy as np
import scipy.io.wavfile

from typing import Tuple, List

import pdb



class SpeechModelTFLiteConverter:

    SAMPLE_RATE = 16000
    WINDOWS_SIZE = 128
    OVERLAP = WINDOWS_SIZE // 2
    EPSILON = 1e-7

    def __init__(self):
        data_dir = pathlib.Path(__file__).parent / "data"
        self.train_file_names_path = data_dir / "train_files.csv"
        self.audio_dir = data_dir / "audio"
        # self.test_files_path = data_dir / "test_files.csv"

    def calculate_log_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Calculates log_spectrogram (frequencies over time plot) of the audio signal.

        Args:
            audio (np.ndarray): audio signal

        Returns:
            log_spectrogram: logarithm of the spectrogram
        """

        number_of_frequencies = int(self.WINDOWS_SIZE / 2) + 1
        window = np.hanning(self.WINDOWS_SIZE)
        time_samples = int(self.SAMPLE_RATE / (self.WINDOWS_SIZE - self.OVERLAP))
        log_spectrogram = np.empty((time_samples - 1, number_of_frequencies))

        for i in range(1, time_samples):
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

    def read_vaw_files(self, files_sample: pd.DataFrame) -> Tuple[np.ndarray, List]:
        """Reads files from the dataframe, pads them and transforms the log_spectrograms
            to ndarrays.

        Args:
            files_sample (pd.DataFrame): DataFrame containing the file names of the audio files,
                mapped to the spoken word in the audio.

        Returns:
            x_array (np.ndarray): Array of size (number of files, 99, 71) containing log_spectrograms
                for each file.
            y_array (np.ndarray): Array of the corresponding classes for each file.
        """

        time_samples = int(self.SAMPLE_RATE / (self.WINDOWS_SIZE - self.OVERLAP))
        resized_overlap = self.OVERLAP + 1
        x_array = np.empty((len(files_sample), time_samples - 1, resized_overlap))
        y_array = []

        for i, row in files_sample.iterrows():
            file_path = self.audio_dir / row["class"] / row["filename"]
            sample_rate, samples = scipy.io.wavfile.read(file_path)
            padded_samples = self.pad_audio(samples)
            spectrogram = self.calculate_log_spectrogram(padded_samples)

            pdb.set_trace()

            # truncate samples that are too long
            if spectrogram.shape[1] > resized_overlap:
                x_array[i, :, :] = spectrogram[:, :resized_overlap]
            else:
                x_array[i, :, :] = spectrogram[:, :]

            y_array.append(row["class"])

        return x_array, y_array

    def read_files(self):
        train_sample_files = pd.read_csv(self.train_file_names_path, index_col=0)
        # test_sample_files = pd.read_csv(self.test_files_path)
        x_train, y_train, = self.read_vaw_files(train_sample_files)

        pdb.set_trace()

    # def prepare_data(self):
    #     pass
    #
    # def conv_1d_time_stacked_model(input_size: int, num_classes: int = 15):
    #     """ Creates a 1D model for temporal data.
    #
    #   Args:
    #     input_size: How big the input vector is.
    #     num_classes: How many classes are to be recognized.
    #   Returns:
    #     Compiled keras model
    #   """
    #     input_layer = Input(shape=input_size)
    #     tens = input_layer
    #
    #     def _reduce_conv(tens: Tensor, num_filters: int, k: int, strides: int = 2, padding: str = "valid"):
    #         """ Building block of the neurat network. It includes the Conv1D layer + normalization+relu  + max pooling
    #
    #         Args:
    #             tens (Tensor):
    #             num_filters (int): Number of filters in the convolutional layer
    #             k (int): The number of positions by which the filter is moved right at each step.
    #             strides (int):  Maxx pooling strides
    #             padding (str):  Type of padding used
    #         Returns:
    #             tens (tf,tensor): tensor after the conv1D layer
    #         """
    #         tens = Conv1D(
    #             num_filters,
    #             k,
    #             padding=padding,
    #             use_bias=False,
    #             kernel_regularizer=l2(0.00002),
    #         )(tens)
    #         tens = BatchNormalization()(tens)
    #         tens = Activation(relu)(tens)
    #         tens = MaxPool1D(pool_size=3, strides=strides, padding=padding)(tens)
    #         return tens
    #
    #     def _context_conv(tens: Tensor, num_filters: int, k: int, dilation_rate: int = 1, padding: str = "valid"):
    #         """ Building block of the neurat network. It includes the Conv1D layer + normalization+relu activation function
    #         Args:
    #             tens (Tensor):
    #             num_filters: Number of filters in the convolutional layer
    #             k (int): The number of positions by which the filter is moved right at each step.
    #             dilation_rate (int):  Number of skipped cells between filters
    #             padding (str):  Type of padding used
    #         Returns:
    #             tens (Tensor): tensor after the conv1D layer
    #         """
    #         tens = Conv1D(
    #             num_filters,
    #             k,
    #             padding=padding,
    #             dilation_rate=dilation_rate,
    #             kernel_regularizer=l2(0.00002),
    #             use_bias=False,
    #         )(tens)
    #         tens = BatchNormalization()(tens)
    #         tens = Activation(relu)(tens)
    #         return tens
    #
    #     tens = _context_conv(tens, 16, 1)
    #     tens = _reduce_conv(tens, 32, 3)
    #     tens = _context_conv(tens, 32, 3)
    #
    #     tens = _reduce_conv(tens, 64, 3)
    #     tens = Dropout(0.1)(tens)
    #     tens = _context_conv(tens, 64, 3)
    #
    #     tens = _reduce_conv(tens, 128, 3)
    #     tens = Dropout(0.1)(tens)
    #     tens = _context_conv(tens, 128, 3)
    #
    #     tens = _reduce_conv(tens, 249, 3)
    #     tens = _context_conv(tens, 249, 3)
    #
    #     tens = Dropout(0.1)(tens)
    #     tens = Conv1D(num_classes, 9, activation="softmax")(tens)
    #     tens = Reshape([-1])(tens)
    #
    #     model = Model(input_layer, tens, name="conv_1d_time_stacked")
    #     model.compile(
    #         loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
    #     )
    #     return model

    def fit_model(self):
        pass

    def convert(self):
        self.read_files()
        # self.prepare_data()
        # self.fit_model()


def main():
    converter = SpeechModelTFLiteConverter()
    converter.convert()


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


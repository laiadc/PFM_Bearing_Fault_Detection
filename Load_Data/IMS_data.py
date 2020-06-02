import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import signal

from sklearn.model_selection import train_test_split


class Import_Data_IMS():
    """
    Reads the IMS data from a csv file and stores the 4 waveforms for each bearing.


    Args:
      file (str): filename of the csv file containing the IMS data

    Attributes:
      t_0 (data frame): Waveform of bearing 1
      t_1 (data frame): Waveform of bearing 2
      t_2 (data frame): Waveform of bearing 3
      t_3 (data frame): Waveform of bearing 4
    """

    def __init__(self, file):
        # Read file
        df_t = pd.read_csv(file)
        # Reformat file
        df_t.drop(df_t.columns[0], axis=1, inplace=True)
        t = df_t.values.astype('float32')
        # Split data into the four bearings waveforms
        ts = np.split(t, 4)

        self.t_0 = ts[0]
        self.t_1 = ts[1]
        self.t_2 = ts[2]
        self.t_3 = ts[3]


class Preprocessing_IMS():
    """
    Preprocesses the IMS dataset to feed the classification model.
    The preprocessing steps are the following:
      1. Downsampling from 20000 Hz to 5000 Hz in order to have data more similiar to the real productio data.
      2. Rolling window for each waveform, to reduce the size of the wavevorm to 1000 timestamps
         (instead of 5000) and to obtain more samples from the same waveform. There is no overlap.
      3. Adaptative normalization: minmax scaling.
      4. Shuffle samples.
      5. Split of training and test sets.


    Args:
      inner (data frame): Data suffering inner ring fault
      outer (data frame): Data suffering outer ring fault
      rolling (data frame): Data suffering rolling element fault
      healthy (data frame): Healthy data

    Attributes:
      inner (data frame): Data suffering inner ring fault
      outer (data frame): Data suffering outer ring fault
      rolling (data frame): Data suffering rolling element fault
      healthy (data frame): Healthy data

    """

    def __init__(self, inner, outer, rolling, healthy):
        self.inner = inner
        self.outer = outer
        self.rolling = rolling
        self.healthy = healthy

    def preprocess(self, data, num_points=5000, step=1000, l=1000):
        """
        Main preprocessing function. It performs the downsampling to 5000 Hz,
        then it applies the rolling window to obtain more data samples.
        Finally, it applies min max scaling (adaptative normalization).

        Args:
          data (data frame): Data frame that we want to preprocess
          num_points (int): Number of timestamps after doing the downsampling
          step (int):  Number of points between two successive waveforms.
                       If step = l, no overlap is applied.
          l (int): Length of the waveforms that will be fed to the model

        Returns:
          prep_data (data frame): Preprocessed data frame
        """
        # Downsampling to more or less 5kHz
        data_dec = signal.decimate(data, 4)

        # Rolling window.
        data_dec = data_dec[:, :5000]
        prep_data = np.empty((0, l))
        for i in np.arange(0, num_points - l + step, step):
            reshaped = data_dec[:, i: i + l].reshape(-1, l)
            # Stand y minmax scalin
            reshaped = preprocessing.scale(reshaped, axis=1)
            reshaped = preprocessing.minmax_scale(reshaped, axis=1)
            prep_data = np.append(prep_data, reshaped, axis=0)

        return prep_data

    def preprocess_y(self, data_y, l=5):
        """
        Gives the vector of labels (y) for the preprocessed data. Since we have increased
        the number of samples by using the rolling window, the number of labels also needs
        to be increased.

        Args:
          data_y (numpy array): Array of labels of the original data frame (before preprocessing)
          l (int): Number of waveforms obtained from the original data. That is, if we have
                   used a rolling window of lenght 1000 from a waveform of length 5000 with
                   no overlap, we have generated 5 new waveforms from the old one. Thus, l=5

        Returns:
          y_prep (data frame): Vector of labels for the preprocessed data
        """
        y_prep = []
        # Repeat the labels from data_y 'l' times
        for i in range(data_y.shape[0]):
            aux = np.repeat(data_y[i], l).tolist()
            y_prep = y_prep + aux

        y_prep = np.asarray(y_prep)
        return y_prep

    def shuffle(self, X_train, X_test, y_train, y_test):
        """
        Shuffles the indices from the train and test sets, so that
        faults and loads are not ordered.

        Args:
          X_train (data frame): Training data (waveforms)
          X_Test (data frame): Test data (waveforms)
          y_train (numpy array): Training labels
          y_test (numpy array): Test labels

        Returns:
          X_train (data frame): Shuffled training data (waveforms)
          X_Test (data frame): Shuffled test data (waveforms)
          y_train (numpy array): Shuffled training labels
          y_test (numpy array): Shuffled test labels
        """
        # Shuffle indeces so that the loads/defects are not ordered
        # Training data
        idx = np.arange(X_train.shape[0])
        np.random.shuffle(idx)
        X_train = X_train[idx, :]
        y_train = y_train[idx]

        # Test data
        idx = np.arange(X_test.shape[0])
        np.random.shuffle(idx)
        X_test = X_test[idx, :]
        y_test = y_test[idx]

        return X_train, X_test, y_train, y_test

    def get_X_y(self):
        """
        Wrapping up function which returns the training and test data applying all
        the preprocessing techniques defined by the previous functions.

        Args:
          -

        Returns:
          X_train (data frame): Processed training data (waveforms)
          X_Test (data frame): Processed test data (waveforms)
          y_train (numpy array): Processed training labels
          y_test (numpy array): Processed test labels
        """
        # Split training and test sets for the healthy, inner, outer and rolling data
        X_train_h, X_test_h, _, _ = train_test_split(self.healthy, np.zeros(self.healthy.shape), test_size=0.33,
                                                     random_state=42)
        X_train_i, X_test_i, _, _ = train_test_split(self.inner, np.zeros(self.inner.shape), test_size=0.33,
                                                     random_state=42)
        X_train_o, X_test_o, _, _ = train_test_split(self.outer, np.zeros(self.outer.shape), test_size=0.33,
                                                     random_state=42)
        X_train_r, X_test_r, _, _ = train_test_split(self.rolling, np.zeros(self.rolling.shape), test_size=0.33,
                                                     random_state=42)

        # Preprocess healthy data
        X_train_h = self.preprocess(X_train_h)
        X_test_h = self.preprocess(X_test_h)

        # Preprocess inner ring data
        X_train_i = self.preprocess(X_train_i)
        X_test_i = self.preprocess(X_test_i)

        # Preprocess outer ring data
        X_train_o = self.preprocess(X_train_o)
        X_test_o = self.preprocess(X_test_o)

        # Preprocess rolling element data
        X_train_r = self.preprocess(X_train_r)
        X_test_r = self.preprocess(X_test_r)

        # Concatenate all the training / test data
        X_train = np.concatenate([X_train_h, X_train_i, X_train_o, X_train_r])
        X_test = np.concatenate([X_test_h, X_test_i, X_test_o, X_test_r])

        # Calculate the labels for the training data
        y_train = np.zeros(X_train.shape[0])
        y_train[X_train_h.shape[0]:(X_train_h.shape[0] + X_train_i.shape[0])] = 1
        y_train[
        (X_train_h.shape[0] + X_train_i.shape[0]):(X_train_h.shape[0] + X_train_i.shape[0] + X_train_o.shape[0])] = 2
        y_train[(X_train_h.shape[0] + X_train_i.shape[0] + X_train_o.shape[0]):] = 3

        # Calculate the lables for the test data
        y_test = np.zeros(X_test.shape[0])
        y_test[X_test_h.shape[0]:(X_test_h.shape[0] + X_test_i.shape[0])] = 1
        y_test[(X_test_h.shape[0] + X_test_i.shape[0]):(X_test_h.shape[0] + X_test_i.shape[0] + X_test_o.shape[0])] = 2
        y_test[(X_test_h.shape[0] + X_test_i.shape[0] + X_test_o.shape[0]):] = 3

        # Shuffle the training and test sets so that the faulty data is not ordered
        X_train, X_test, y_train, y_test = self.shuffle(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test


class Preprocessing_IMS_AE:
    """
    Preprocesses the IMS dataset to feed the AE model.
    The preprocessing steps are the following:
      1. Downsampling from 20000 Hz to 5000 Hz in order to have data more similiar to the real productio data.
      2. Rolling window for each waveform, to reduce the size of the wavevorm to 1000 timestamps
         (instead of 5000) and to obtain more samples from the same waveform. There is no overlap.
      3. Adaptative normalization: minmax scaling.
      4. Shuffle samples.
      5. Split of training and test sets.


    Args:
      healthy (data frame): Healthy data
      num_points (int): Number of timestamps after doing the downsampling
      step (int):  Number of points between two successive waveforms.
                     If step = l, no overlap is applied.
      l (int): Length of the waveforms that will be fed to the model

    Attributes:
      healthy (data frame): Healthy data
      num_points (int): Number of timestamps after doing the downsampling
      step (int):  Number of points between two successive waveforms.
                     If step = l, no overlap is applied.
      l (int): Length of the waveforms that will be fed to the model

    """

    def __init__(self, normal, num_points=5000, step=100, l=1000):
        self.normal = normal
        self.num_points = num_points
        self.step = step
        self.l = l

    def preprocess(self, data, num_points=5000, step=100, l=1000):
        """
        Main preprocessing function. It performs the downsampling to 5000 Hz,
        then it applies the rolling window to obtain more data samples.
        Finally, it applies min max scaling (adaptative normalization).

        Args:
          data (data frame): Data frame that we want to preprocess
          num_points (int): Number of timestamps after doing the downsampling
          step (int):  Number of points between two successive waveforms.
                       If step = l, no overlap is applied.
          l (int): Length of the waveforms that will be fed to the model

        Returns:
          prep_data (data frame): Preprocessed data frame

        """
        # Downsampling to more or less 5kHz
        data_dec = signal.decimate(data, 4)

        # Rolling window.
        data_dec = data_dec[:, :5000]
        prep_data = np.empty((0, l))
        for i in np.arange(0, num_points - l + step, step):
            reshaped = data_dec[:, i: i + l].reshape(-1, l)
            prep_data = np.append(prep_data, reshaped, axis=0)

        # Stand y minmax scalin
        prep_data = preprocessing.scale(prep_data.T).T
        prep_data = preprocessing.minmax_scale(prep_data.T).T
        return prep_data

    def preprocess_test(self, data):
        """
        Preprocesses data for testing (Calculating the reconstructon error/output of the model)

        Args:
          data (numpy array): Data to process

        Returns:
          min_maxed (numpy array): Preprocessed data for testing
        """
        # Downsampling to 5000 Hz
        data = signal.decimate(data, 4)[:, :4000]
        # Minmax scaling
        scaled = preprocessing.scale(data.T).T
        min_maxed = preprocessing.minmax_scale(scaled.T).T
        return min_maxed

    def shuffle(self, X_train, X_test):
        """
        Shuffles the indices from the train and test sets, so that
        faults and loads are not ordered.

        Args:
          X_train (data frame): Training data (waveforms)
          X_Test (data frame): Test data (waveforms)

        Returns:
          X_train (data frame): Shuffled training data (waveforms)
          X_Test (data frame): Shuffled test data (waveforms)
        """
        # Shuffle indeces so that the loads/defects are not ordered
        # Training data
        idx = np.arange(X_train.shape[0])
        np.random.shuffle(idx)
        X_train = X_train[idx, :]

        # Test data
        idx = np.arange(X_test.shape[0])
        np.random.shuffle(idx)
        X_test = X_test[idx, :]

        return X_train, X_test

    def get_X_y(self):
        """
        Get training and test IMS data
        Args:
            -
        Returns:
          X_train (data frame): Training data
          X_test (data_frame): Test data
        """
        train_test = self.preprocess(self.normal, self.num_points, self.step, self.l)

        X_train, X_test, _, _ = train_test_split(train_test, train_test, test_size=0.33, random_state=42)

        X_train, X_test = self.shuffle(X_train, X_test)

        return X_train, X_test

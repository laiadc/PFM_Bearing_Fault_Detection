import numpy as np
import pandas as pd
import scipy
import scipy.signal
import scipy.io
from sklearn.model_selection import train_test_split


class Preprocessing_CWRU():
    """
    Preprocessing functions for either Paderborn/CWRU datasets.
    The preprocessing steps are the following:
      1. Downsampling from 20000 Hz to 5000 Hz in order to have data more similiar to the real productio data.
      2. Rolling window for each waveform, to reduce the size of the wavevorm to 1000 timestamps
         (instead of 5000) and to obtain more samples from the same waveform. There is no overlap.
      3. Adaptative normalization: minmax scaling.
      4. Shuffle samples.
      5. Split of training and test sets.

    Args:
      -
    Returns:
      -
    """

    def __init__(self):
        pass

    def normalize(self, x, high=1., low=-1.):
        """
        Applies min-max scaling (adaptative normalization)

        Args:
          x (numpy array): data to normalize
          high (float): maximum value of the normalized data
          low (float): minimum value of the normalized data

        Returns:
          normalized (numpy array): normalized data

        """
        max_val = np.max(x)
        min_val = np.min(x)
        normalized = (x - min_val) / max_val + low

        return (normalized)

    def add_noise_band(self, data, f_mean, f_std, A_mean, A_std, num_noise, time):
        """
        Adds random periodic noise with amplitude and frequency following a normal
        distribution

        Args:
          data (numpy array): Data to add noise to
          f_mean (float): Mean frequency of the added noise
          f_std (float): Standard deviation of the added noise
          A_mean (float): Mean amplitude of the added noise
          A_std (float): Standard deviation of the added noise
          num_noise (int): Number of random samples to add to the original signal
          time (float): Duration of the signal

        Returns:
          data (numpy array): Data with added noise
        """
        # Generate time range
        t = np.arange(0.0, time, time / data.shape[0])
        data_prep = np.copy(data)
        # Generate random samples of noise and add them to the original signal
        for num in range(num_noise):
            A = np.abs(np.random.normal(A_mean, A_std))  # Generate random amplitude
            f = np.random.normal(f_mean, f_std)  # Generate random frequency
            f = f / 60.
            wave = A * np.sin(2 * np.pi * f * t)  # Generate noise signal
            leng = np.min([wave.shape[0], data.shape[0]])
            wave = wave[0:leng]
            data_prep = data_prep[0:leng]
            data_prep += wave  # Add noise to signal

        return data_prep

    def splitData(self, data, defect, load, length=1000, samp_f=5000, overlap=0, A_m=0, A_sd=0.08, f_m=300000,
                  f_sd=20000, noise=True):
        """
        Splits data to match the input size of the model and adds random noise within a frequency band
        Args:
          data (numpy array): Waveforms to preprocess
          defect (numpy array): Type of failure of the bearing (0-> Healthy, 1-> Inner ring, 2-> Outer ring)
          load (numpy array): Type of load of the machine (0-> 900 Hz, 1-> 1500 Hz)
          length (int): Size of the waveform for the model
          samp_f (int): Desired sampling frequency
          overlap (int): Windows overlapping
          f_m (float): Mean frequency of the added noise
          f_sd (float): Standard deviation of the added noise
          A_m (float): Mean amplitude of the added noise
          A_sd (float): Standard deviation of the added noise
          noise (bool): Wether or not to add noise

        Returns:
          data_prep (data frame): Processed data
        """
        # Create array of preprocessed data
        data_prep = np.zeros((len(range(0, len(data), length - overlap)) - 1, length))
        j = 0
        for i in range(0, len(data), length - overlap):
            # Split waveforms with rolling windows
            time = length / samp_f
            if len(data[i:i + length]) == length:
                if noise:  # Add noise to bandwith
                    data_prep[j, :] = self.add_noise_band(data[i:i + length], f_m, f_sd, A_m, A_sd, 10, time)
                else:
                    data_prep[j, :] = data[i:i + length]
                j += 1
        # Convert to data frame
        data_prep = pd.DataFrame(data_prep)
        data_prep.columns = ["t" + str(i) for i in range(length)]
        # Min-max normalization
        data_prep = data_prep.apply(lambda x: self.normalize(x), axis=1)
        # Add load and Defect columns
        data_prep['Defect'] = defect
        data_prep['Load'] = load
        return data_prep

    def add_noise_harmonics(self, data, f):
        """
        Amplifies the frequency of the first harmonics to add noise.

        Args:
          data (data frame): Data to add noise to
          f (float): Rotation velocity

        Returns:
          data_prep (data frame): Data with noise
        """
        if f <= 1500:  # This data belongs to Paderborn
            time = np.arange(0.0, 4.0, 4. / data.shape[0])  # Time axis to add noise
        else:
            t = data.shape[0] / 12000  # t max
            time = np.arange(0.0, t, 1. / 12000)

        # Calculate frequency of first harmonics (in Hz)
        f1 = 2.0 * f / 60.
        f2 = 3.0 * f / 60.
        f3 = 4.0 * f / 60.

        data_prep = np.copy(data)
        # For each frequency, add the signal with that frequency and random amplitude
        for fi in [f1, f2, f3]:
            A = np.random.uniform(low=0.02, high=0.04)  # Select random amplitude
            wave = A * np.sin(2 * np.pi * fi * time)  # Generate waveform of noise
            leng = np.min([wave.shape[0], data.shape[0]])
            wave = wave[0:leng]  # Crop it to fit the data length
            data_prep = data_prep[0:leng]
            data_prep += wave

        return data_prep


class Load_CWRU_noise():
    """
    CLass to load Case Western Reserve University data with noise. It uses the
    preprocessing functions from the preprocessing class.

    Args:
       preprocessing (Preprocessing): Preprocessing object

    Attributes:
      healthy0 (data frame): Healthy data from first load
      healthy1 (data frame): Healthy data from second load
      healthy2 (data frame): Healthy data from third load
      healthy3 (data frame): Healthy data from fourth load

      inner0 (data frame): Inner ring data from first load
      inner1 (data frame): Inner ring data from second load
      inner2 (data frame): Inner ring data from third load
      inner3 (data frame): Inner ring data from fourth load

      outer0 (data frame): Outer ring data from first load
      outer1 (data frame): Outer ring data from second load
      outer2 (data frame): Outer ring data from third load
      outer3 (data frame): Outer ring data from fourth load

    """

    def __init__(self, preprocessing):
        self.preprocessing = preprocessing
        # Load data, downsample and add noise
        # We load data with two different diameter failures
        healthy0_raw = scipy.io.loadmat('Data/CWRU/D7/Normal/97.mat')['X097_DE_time'].reshape(1, -1)[0]  # Load
        healthy0_raw = scipy.signal.decimate(healthy0_raw, 2)  # Downsample to 5000 Hz
        healthy0_raw = self.preprocessing.add_noise_harmonics(healthy0_raw, 1797)  # Add noise

        healthy1_raw = scipy.io.loadmat('Data/CWRU/D7/Normal/98.mat')['X098_DE_time'].reshape(1, -1)[0]
        healthy1_raw = scipy.signal.decimate(healthy1_raw, 2)
        healthy1_raw = self.preprocessing.add_noise_harmonics(healthy1_raw, 1772)

        healthy2_raw = scipy.io.loadmat('Data/CWRU/D7/Normal/99.mat')['X099_DE_time'].reshape(1, -1)[0]
        healthy2_raw = scipy.signal.decimate(healthy2_raw, 2)
        healthy2_raw = self.preprocessing.add_noise_harmonics(healthy2_raw, 1750)

        healthy3_raw = scipy.io.loadmat('Data/CWRU/D7/Normal/100.mat')['X100_DE_time'].reshape(1, -1)[0]
        healthy3_raw = scipy.signal.decimate(healthy3_raw, 2)
        healthy3_raw = self.preprocessing.add_noise_harmonics(healthy3_raw, 1730)

        inner0_raw = scipy.io.loadmat('Data/CWRU/D7/Inner/105.mat')['X105_DE_time'].reshape(1, -1)[0]
        inner0_raw = scipy.signal.decimate(inner0_raw, 2)
        inner0_raw = self.preprocessing.add_noise_harmonics(inner0_raw, 1797)

        inner1_raw = scipy.io.loadmat('Data/CWRU/D7/Inner/106.mat')['X106_DE_time'].reshape(1, -1)[0]
        inner1_raw = scipy.signal.decimate(inner1_raw, 2)
        inner1_raw = self.preprocessing.add_noise_harmonics(inner1_raw, 1772)

        inner2_raw = scipy.io.loadmat('Data/CWRU/D7/Inner/107.mat')['X107_DE_time'].reshape(1, -1)[0]
        inner2_raw = scipy.signal.decimate(inner2_raw, 2)
        inner2_raw = self.preprocessing.add_noise_harmonics(inner2_raw, 1750)

        inner3_raw = scipy.io.loadmat('Data/CWRU/D7/Inner/108.mat')['X108_DE_time'].reshape(1, -1)[0]
        inner3_raw = scipy.signal.decimate(inner3_raw, 2)
        inner3_raw = self.preprocessing.add_noise_harmonics(inner3_raw, 1730)

        outer0_raw = scipy.io.loadmat('Data/CWRU/D7/Outer/130.mat')['X130_DE_time'].reshape(1, -1)[0]
        outer0_raw = scipy.signal.decimate(outer0_raw, 2)
        outer0_raw = self.preprocessing.add_noise_harmonics(outer0_raw, 1797)

        outer1_raw = scipy.io.loadmat('Data/CWRU/D7/Outer/131.mat')['X131_DE_time'].reshape(1, -1)[0]
        outer1_raw = scipy.signal.decimate(outer1_raw, 2)
        outer1_raw = self.preprocessing.add_noise_harmonics(outer1_raw, 1772)

        outer2_raw = scipy.io.loadmat('Data/CWRU/D7/Outer/132.mat')['X132_DE_time'].reshape(1, -1)[0]
        outer2_raw = scipy.signal.decimate(outer2_raw, 2)
        outer2_raw = self.preprocessing.add_noise_harmonics(outer2_raw, 1750)

        outer3_raw = scipy.io.loadmat('Data/CWRU/D7/Outer/133.mat')['X133_DE_time'].reshape(1, -1)[0]
        outer3_raw = scipy.signal.decimate(outer3_raw, 2)
        outer3_raw = self.preprocessing.add_noise_harmonics(outer3_raw, 1750)

        inner0_raw2 = scipy.io.loadmat('Data/CWRU/D14/Inner/169.mat')['X169_DE_time'].reshape(1, -1)[0]
        inner0_raw2 = scipy.signal.decimate(inner0_raw2, 2)
        inner0_raw2 = self.preprocessing.add_noise_harmonics(inner0_raw2, 1797)

        inner1_raw2 = scipy.io.loadmat('Data/CWRU/D14/Inner/170.mat')['X170_DE_time'].reshape(1, -1)[0]
        inner1_raw2 = scipy.signal.decimate(inner1_raw2, 2)
        inner1_raw2 = self.preprocessing.add_noise_harmonics(inner1_raw2, 1772)

        inner2_raw2 = scipy.io.loadmat('Data/CWRU/D14/Inner/171.mat')['X171_DE_time'].reshape(1, -1)[0]
        inner2_raw2 = scipy.signal.decimate(inner2_raw2, 2)
        inner2_raw2 = self.preprocessing.add_noise_harmonics(inner2_raw2, 1750)

        inner3_raw2 = scipy.io.loadmat('Data/CWRU/D14/Inner/172.mat')['X172_DE_time'].reshape(1, -1)[0]
        inner3_raw2 = scipy.signal.decimate(inner3_raw2, 2)
        inner3_raw2 = self.preprocessing.add_noise_harmonics(inner3_raw2, 1730)

        outer0_raw2 = scipy.io.loadmat('Data/CWRU/D14/Outer/197.mat')['X197_DE_time'].reshape(1, -1)[0]
        outer0_raw2 = scipy.signal.decimate(outer0_raw2, 2)
        outer0_raw2 = self.preprocessing.add_noise_harmonics(outer0_raw2, 1797)

        outer1_raw2 = scipy.io.loadmat('Data/CWRU/D14/Outer/198.mat')['X198_DE_time'].reshape(1, -1)[0]
        outer1_raw2 = scipy.signal.decimate(outer1_raw2, 2)
        outer1_raw2 = self.preprocessing.add_noise_harmonics(outer1_raw2, 1772)

        outer2_raw2 = scipy.io.loadmat('Data/CWRU/D14/Outer/199.mat')['X199_DE_time'].reshape(1, -1)[0]
        outer2_raw2 = scipy.signal.decimate(outer2_raw2, 2)
        outer2_raw2 = self.preprocessing.add_noise_harmonics(outer2_raw2, 1750)

        outer3_raw2 = scipy.io.loadmat('Data/CWRU/D14/Outer/200.mat')['X200_DE_time'].reshape(1, -1)[0]
        outer3_raw2 = scipy.signal.decimate(outer3_raw2, 2)
        outer3_raw2 = self.preprocessing.add_noise_harmonics(outer3_raw2, 1730)

        # Split data with rolling windows
        self.healthy0 = self.preprocessing.splitData(healthy0_raw, 0, 0)
        self.healthy1 = self.preprocessing.splitData(healthy1_raw, 0, 1)
        self.healthy2 = self.preprocessing.splitData(healthy2_raw, 0, 2)
        self.healthy3 = self.preprocessing.splitData(healthy3_raw, 0, 3)

        inner0 = self.preprocessing.splitData(inner0_raw, 1, 0)
        inner1 = self.preprocessing.splitData(inner1_raw, 1, 1)
        inner2 = self.preprocessing.splitData(inner2_raw, 1, 2)
        inner3 = self.preprocessing.splitData(inner3_raw, 1, 3)

        outer0 = self.preprocessing.splitData(outer0_raw, 2, 0)
        outer1 = self.preprocessing.splitData(outer1_raw, 2, 1)
        outer2 = self.preprocessing.splitData(outer2_raw, 2, 2)
        outer3 = self.preprocessing.splitData(outer3_raw, 2, 3)

        inner02 = self.preprocessing.splitData(inner0_raw2, 1, 0)
        inner12 = self.preprocessing.splitData(inner1_raw2, 1, 1)
        inner22 = self.preprocessing.splitData(inner2_raw2, 1, 2)
        inner32 = self.preprocessing.splitData(inner3_raw2, 1, 3)

        outer02 = self.preprocessing.splitData(outer0_raw2, 2, 0)
        outer12 = self.preprocessing.splitData(outer1_raw2, 2, 1)
        outer22 = self.preprocessing.splitData(outer2_raw2, 2, 2)
        outer32 = self.preprocessing.splitData(outer3_raw2, 2, 3)

        # Concatenate data from different diameters failures
        self.inner0 = pd.concat([inner0, inner02])
        self.inner1 = pd.concat([inner1, inner12])
        self.inner2 = pd.concat([inner2, inner22])
        self.inner3 = pd.concat([inner3, inner32])

        self.outer0 = pd.concat([outer0, outer02])
        self.outer1 = pd.concat([outer1, outer12])
        self.outer2 = pd.concat([outer2, outer22])
        self.outer3 = pd.concat([outer3, outer32])

    def CreateDataSets(self, listdf, length=1000):
        """
        Prepares training and test datasets, applying the processing functions.

        Args:
          listdf (list of data frames): List containig all the datasets to be added to the training/test sets
          lenght (int): Input length for the model

        Returns:
          X_train (numpy array): Processed X train data
          y_train (numpy array): Labels of X_train
          load_train (numpy array): Labels of the velocities/loads for X_train
          X_test (numpy array): Processed X test data
          y_test (numpy array): Labels of X_test
          load_test (numpy array): Labels of the velocities/loads for X_test

        """
        # Prepare data frames and arrays
        X_train = pd.DataFrame(np.zeros((1, length)))
        X_train.columns = ["t" + str(i) for i in range(length)]
        X_test = pd.DataFrame(np.zeros((1, length)))
        X_test.columns = ["t" + str(i) for i in range(length)]
        y_train = np.array([])
        y_test = np.array([])
        load_train = np.array([])
        load_test = np.array([])

        # Preprocess each data frame from the list
        for df in listdf:
            X = df.drop("Defect", axis=1)
            y = df["Defect"]
            # Split train/test
            X_traindf, X_testdf, y_traindf, y_testdf = train_test_split(X, y, test_size=0.25, random_state=42)

            load_traindf = X_traindf["Load"]
            X_traindf.drop("Load", inplace=True, axis=1)
            load_testdf = X_testdf["Load"]
            X_testdf.drop("Load", inplace=True, axis=1)

            X_train = pd.concat([X_train, X_traindf], axis=0)
            X_test = pd.concat([X_test, X_testdf])

            y_train = np.append(y_train, y_traindf)
            y_test = np.append(y_test, y_testdf)

            load_train = np.append(load_train, load_traindf)
            load_test = np.append(load_test, load_testdf)

        X_train = X_train.values[1:, :]
        X_test = X_test.values[1:, :]

        return X_train, X_test, y_train, y_test, load_train, load_test

    def get_X_y(self):
        """
        Wrap-up function to generate CWRU training and test datasets

        Args:
          -
        Returns:
          X_train (numpy array): Processed X train data
          y_train (numpy array): Labels of X_train
          load_train (numpy array): Labels of the velocities/loads for X_train
          X_test (numpy array): Processed X test data
          y_test (numpy array): Labels of X_test
          load_test (numpy array): Labels of the velocities/loads for X_test
        """
        listdf = [self.healthy0, self.healthy1, self.healthy2, self.healthy3,
                  self.inner0, self.inner1, self.inner2, self.inner3,
                  self.outer0, self.outer1, self.outer2, self.outer3]

        X_train_CWRU, X_test_CWRU, y_train_CWRU, y_test_CWRU, load_train_CWRU, load_test_CWRU = self.CreateDataSets(
            listdf, length=1000)

        return X_train_CWRU, X_test_CWRU, y_train_CWRU, y_test_CWRU, load_train_CWRU, load_test_CWRU


class Load_CWRU():
    """
    Class to load Case Western Reserve University data without noise. It uses the
    preprocessing functions from the preprocessing class.

    Args:
       preprocessing (Preprocessing): Preprocessing object

    Attributes:
      healthy0 (data frame): Healthy data from first load
      healthy1 (data frame): Healthy data from second load
      healthy2 (data frame): Healthy data from third load
      healthy3 (data frame): Healthy data from fourth load

      inner0 (data frame): Inner ring data from first load
      inner1 (data frame): Inner ring data from second load
      inner2 (data frame): Inner ring data from third load
      inner3 (data frame): Inner ring data from fourth load

      outer0 (data frame): Outer ring data from first load
      outer1 (data frame): Outer ring data from second load
      outer2 (data frame): Outer ring data from third load
      outer3 (data frame): Outer ring data from fourth load

    """

    def __init__(self, preprocessing):
        self.preprocessing = preprocessing
        # Load data, downsample
        healthy0_raw = scipy.io.loadmat('Data/CWRU/D7/Normal/97.mat')['X097_DE_time'].reshape(1, -1)[0]
        healthy0_raw = scipy.signal.decimate(healthy0_raw, 2)

        healthy1_raw = scipy.io.loadmat('Data/CWRU/D7/Normal/98.mat')['X098_DE_time'].reshape(1, -1)[0]
        healthy1_raw = scipy.signal.decimate(healthy1_raw, 2)

        healthy2_raw = scipy.io.loadmat('Data/CWRU/D7/Normal/99.mat')['X099_DE_time'].reshape(1, -1)[0]
        healthy2_raw = scipy.signal.decimate(healthy2_raw, 2)

        healthy3_raw = scipy.io.loadmat('Data/CWRU/D7/Normal/100.mat')['X100_DE_time'].reshape(1, -1)[0]
        healthy3_raw = scipy.signal.decimate(healthy3_raw, 2)

        inner0_raw = scipy.io.loadmat('Data/CWRU/D7/Inner/105.mat')['X105_DE_time'].reshape(1, -1)[0]
        inner0_raw = scipy.signal.decimate(inner0_raw, 2)

        inner1_raw = scipy.io.loadmat('Data/CWRU/D7/Inner/106.mat')['X106_DE_time'].reshape(1, -1)[0]
        inner1_raw = scipy.signal.decimate(inner1_raw, 2)

        inner2_raw = scipy.io.loadmat('Data/CWRU/D7/Inner/107.mat')['X107_DE_time'].reshape(1, -1)[0]
        inner2_raw = scipy.signal.decimate(inner2_raw, 2)

        inner3_raw = scipy.io.loadmat('Data/CWRU/D7/Inner/108.mat')['X108_DE_time'].reshape(1, -1)[0]
        inner3_raw = scipy.signal.decimate(inner3_raw, 2)

        outer0_raw = scipy.io.loadmat('Data/CWRU/D7/Outer/130.mat')['X130_DE_time'].reshape(1, -1)[0]
        outer0_raw = scipy.signal.decimate(outer0_raw, 2)

        outer1_raw = scipy.io.loadmat('Data/CWRU/D7/Outer/131.mat')['X131_DE_time'].reshape(1, -1)[0]
        outer1_raw = scipy.signal.decimate(outer1_raw, 2)

        outer2_raw = scipy.io.loadmat('Data/CWRU/D7/Outer/132.mat')['X132_DE_time'].reshape(1, -1)[0]
        outer2_raw = scipy.signal.decimate(outer2_raw, 2)

        outer3_raw = scipy.io.loadmat('Data/CWRU/D7/Outer/133.mat')['X133_DE_time'].reshape(1, -1)[0]
        outer3_raw = scipy.signal.decimate(outer3_raw, 2)

        inner0_raw2 = scipy.io.loadmat('Data/CWRU/D14/Inner/169.mat')['X169_DE_time'].reshape(1, -1)[0]
        inner0_raw2 = scipy.signal.decimate(inner0_raw2, 2)

        inner1_raw2 = scipy.io.loadmat('Data/CWRU/D14/Inner/170.mat')['X170_DE_time'].reshape(1, -1)[0]
        inner1_raw2 = scipy.signal.decimate(inner1_raw2, 2)

        inner2_raw2 = scipy.io.loadmat('Data/CWRU/D14/Inner/171.mat')['X171_DE_time'].reshape(1, -1)[0]
        inner2_raw2 = scipy.signal.decimate(inner2_raw2, 2)

        inner3_raw2 = scipy.io.loadmat('Data/CWRU/D14/Inner/172.mat')['X172_DE_time'].reshape(1, -1)[0]
        inner3_raw2 = scipy.signal.decimate(inner3_raw2, 2)

        outer0_raw2 = scipy.io.loadmat('Data/CWRU/D14/Outer/197.mat')['X197_DE_time'].reshape(1, -1)[0]
        outer0_raw2 = scipy.signal.decimate(outer0_raw2, 2)

        outer1_raw2 = scipy.io.loadmat('Data/CWRU/D14/Outer/198.mat')['X198_DE_time'].reshape(1, -1)[0]
        outer1_raw2 = scipy.signal.decimate(outer1_raw2, 2)

        outer2_raw2 = scipy.io.loadmat('Data/CWRU/D14/Outer/199.mat')['X199_DE_time'].reshape(1, -1)[0]
        outer2_raw2 = scipy.signal.decimate(outer2_raw2, 2)

        outer3_raw2 = scipy.io.loadmat('Data/CWRU/D14/Outer/200.mat')['X200_DE_time'].reshape(1, -1)[0]
        outer3_raw2 = scipy.signal.decimate(outer3_raw2, 2)

        # Split data with rolling windows (without noise)
        self.healthy0 = self.preprocessing.splitData(healthy0_raw, 0, 0, noise=False)
        self.healthy1 = self.preprocessing.splitData(healthy1_raw, 0, 1, noise=False)
        self.healthy2 = self.preprocessing.splitData(healthy2_raw, 0, 2, noise=False)
        self.healthy3 = self.preprocessing.splitData(healthy3_raw, 0, 3, noise=False)

        inner0 = self.preprocessing.splitData(inner0_raw, 1, 0, noise=False)
        inner1 = self.preprocessing.splitData(inner1_raw, 1, 1, noise=False)
        inner2 = self.preprocessing.splitData(inner2_raw, 1, 2, noise=False)
        inner3 = self.preprocessing.splitData(inner3_raw, 1, 3, noise=False)

        outer0 = self.preprocessing.splitData(outer0_raw, 2, 0, noise=False)
        outer1 = self.preprocessing.splitData(outer1_raw, 2, 1, noise=False)
        outer2 = self.preprocessing.splitData(outer2_raw, 2, 2, noise=False)
        outer3 = self.preprocessing.splitData(outer3_raw, 2, 3, noise=False)

        inner02 = self.preprocessing.splitData(inner0_raw2, 1, 0, noise=False)
        inner12 = self.preprocessing.splitData(inner1_raw2, 1, 1, noise=False)
        inner22 = self.preprocessing.splitData(inner2_raw2, 1, 2, noise=False)
        inner32 = self.preprocessing.splitData(inner3_raw2, 1, 3, noise=False)

        outer02 = self.preprocessing.splitData(outer0_raw2, 2, 0, noise=False)
        outer12 = self.preprocessing.splitData(outer1_raw2, 2, 1, noise=False)
        outer22 = self.preprocessing.splitData(outer2_raw2, 2, 2, noise=False)
        outer32 = self.preprocessing.splitData(outer3_raw2, 2, 3, noise=False)

        # Concatenate data from different diameters failures
        self.inner0 = pd.concat([inner0, inner02])
        self.inner1 = pd.concat([inner1, inner12])
        self.inner2 = pd.concat([inner2, inner22])
        self.inner3 = pd.concat([inner3, inner32])

        self.outer0 = pd.concat([outer0, outer02])
        self.outer1 = pd.concat([outer1, outer12])
        self.outer2 = pd.concat([outer2, outer22])
        self.outer3 = pd.concat([outer3, outer32])

    def CreateDataSets(self, listdf, length=1000):
        """
        Prepares training and test datasets, applying the processing functions.

        Args:
          listdf (list of data frames): List containig all the datasets to be added to the training/test sets
          lenght (int): Input length for the model

        Returns:
          X_train (numpy array): Processed X train data
          y_train (numpy array): Labels of X_train
          load_train (numpy array): Labels of the velocities/loads for X_train
          X_test (numpy array): Processed X test data
          y_test (numpy array): Labels of X_test
          load_test (numpy array): Labels of the velocities/loads for X_test

        """
        # Prepare data frames and arrays
        X_train = pd.DataFrame(np.zeros((1, length)))
        X_train.columns = ["t" + str(i) for i in range(length)]
        X_test = pd.DataFrame(np.zeros((1, length)))
        X_test.columns = ["t" + str(i) for i in range(length)]
        y_train = np.array([])
        y_test = np.array([])
        load_train = np.array([])
        load_test = np.array([])

        # Preprocess each data frame from the list
        for df in listdf:
            X = df.drop("Defect", axis=1)
            y = df["Defect"]
            # Split train/test
            X_traindf, X_testdf, y_traindf, y_testdf = train_test_split(X, y, test_size=0.25, random_state=42)

            load_traindf = X_traindf["Load"]
            X_traindf.drop("Load", inplace=True, axis=1)
            load_testdf = X_testdf["Load"]
            X_testdf.drop("Load", inplace=True, axis=1)

            X_train = pd.concat([X_train, X_traindf], axis=0)
            X_test = pd.concat([X_test, X_testdf])

            y_train = np.append(y_train, y_traindf)
            y_test = np.append(y_test, y_testdf)

            load_train = np.append(load_train, load_traindf)
            load_test = np.append(load_test, load_testdf)

        X_train = X_train.values[1:, :]
        X_test = X_test.values[1:, :]

        return X_train, X_test, y_train, y_test, load_train, load_test

    def get_X_y(self):
        """
        Wrap-up function to generate CWRU training and test datasets

        Args:
          -
        Returns:
          X_train (numpy array): Processed X train data
          y_train (numpy array): Labels of X_train
          load_train (numpy array): Labels of the velocities/loads for X_train
          X_test (numpy array): Processed X test data
          y_test (numpy array): Labels of X_test
          load_test (numpy array): Labels of the velocities/loads for X_test
        """
        listdf = [self.healthy0, self.healthy1, self.healthy2, self.healthy3,
                  self.inner0, self.inner1, self.inner2, self.inner3,
                  self.outer0, self.outer1, self.outer2, self.outer3]

        X_train_CWRU, X_test_CWRU, y_train_CWRU, y_test_CWRU, load_train_CWRU, load_test_CWRU = self.CreateDataSets(
            listdf, length=1000)

        return X_train_CWRU, X_test_CWRU, y_train_CWRU, y_test_CWRU, load_train_CWRU, load_test_CWRU

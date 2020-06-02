import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy
import scipy.io


class Import_Data_Pad:
    """
    Class to import the Paderborn dataset
    Args:
        DATA_HEALTHY15 (str): Path to healthy data of 1500 rpm
        DATA_HEALTHY09 (str): Path of healthy data of 900 rpm
        DATA_INNER15 (str): Path to inner ring data of 1500 rpm
        DATA_INNER09 (str): Path of inner ring data of 900 rpm
        DATA_OUTER15 (str): Path to outer ring data of 1500 rpm
        DATA_OUTER09 (str): Path of outer ring data of 900 rpm
    Attributes:
        healthy_files15 (list of str): List of healthy filepaths of 1500 rpm
        healthy_files09 (list of str): List of healthy filepaths of 900 rpm
        inner_files15 (list of str): List of inner ring filepaths of 1500 rpm
        inner_files09 (list of str): List of inner ring filepaths of 900 rpm
        outer_files15 (list of str): List of outer ring filepaths of 1500 rpm
        outer_files09 (list of str): List of outer ring filepaths of 900 rpm
    """

    def __init__(self, DATA_HEALTHY15, DATA_HEALTHY09, DATA_INNER15, DATA_INNER09, DATA_OUTER15, DATA_OUTER09):
        self.healthy_files15 = [DATA_HEALTHY15 + f for f in listdir(DATA_HEALTHY15) if isfile(join(DATA_HEALTHY15, f))]
        self.healthy_files09 = [DATA_HEALTHY09 + f for f in listdir(DATA_HEALTHY09) if isfile(join(DATA_HEALTHY09, f))]
        self.inner_files15 = [DATA_INNER15 + f for f in listdir(DATA_INNER15) if isfile(join(DATA_INNER15, f))]
        self.inner_files09 = [DATA_INNER09 + f for f in listdir(DATA_INNER09) if isfile(join(DATA_INNER09, f))]
        self.outer_files15 = [DATA_OUTER15 + f for f in listdir(DATA_OUTER15) if isfile(join(DATA_OUTER15, f))]
        self.outer_files09 = [DATA_OUTER09 + f for f in listdir(DATA_OUTER09) if isfile(join(DATA_OUTER09, f))]


class Preprocessing_Pad:
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

    def add_noise_band(self, file, f_mean, f_std, A_mean, A_std, num_noise, time):
        """
        Adds random periodic noise with amplitude and frequency following a normal
        distribution

        Args:
          file (numpy array): Data to add noise to
          f_mean (float): Mean frequency of the added noise
          f_std (float): Standard deviation of the added noise
          A_mean (float): Mean amplitude of the added noise
          A_std (float): Standard deviation of the added noise
          num_noise (int): Number of random samples to add to the original signal
          time (float): Duration of the signal

        Returns:
          file (numpy array): Data with added noise
        """
        # Generate time range
        t = np.arange(0.0, time, time / file.shape[0])
        # Generate random samples of noise and add them to the original signal
        for num in range(num_noise):
            A = np.abs(np.random.normal(A_mean, A_std))  # Generate random amplitude
            f = np.random.normal(f_mean, f_std)  # Generate random frequency
            f = f / 60.
            wave = A * np.sin(2 * np.pi * f * t)  # Generate noise signal
            leng = np.min([wave.shape[0], file.shape[0]])
            wave = wave[0:leng]
            file = file[0:leng]
            file += wave  # Add noise to signal

        return file

    def splitData(self, data, defect, load, length=1000, samp_f=5000,
                  overlap=0, A_m=0, A_sd=0.05, f_m=300000, f_sd=20000, noise=True):
        """
        Splits data to match the input size of the model and adds random noise within a frequency band
        Args:
          data (numpy array): Waveforms to preprocess
          defect (numpy array): Type of failure of the bearing (0-> Healthy, 1-> Inner ring, 2-> Outer ring)
          load (numpy array): Type of load of the machine (0-> 900 Hz, 1-> 1500 Hz)
          length (int): Size of the waveform for the model
          sampl_f (int): Desired sampling frequency
          overlap (int): Windows overlapping
          f_m (float): Mean frequency of the added noise
          f_sd (float): Standard deviation of the added noise
          A_m (float): Mean amplitude of the added noise
          A_sd (float): Standard deviation of the added noise

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
                # Add white gaussian noise to a bandwidth
                if noise:
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
          data (data frame): Data with noise
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

        # For each frequency, add the signal with that frequency and random amplitude
        for fi in [f1, f2, f3]:
            A = np.random.uniform(low=0.04, high=0.06)  # Select random amplitude
            wave = A * np.sin(2 * np.pi * fi * time)  # Generate waveform of noise
            leng = np.min([wave.shape[0], data.shape[0]])
            wave = wave[0:leng]  # Crop it to fit the data length
            data = data[0:leng]
            data += wave

        return data


class Load_Pad:
    """
    Loads Paderborn data

    Args:
      preproccesing (Preprocessing): Object containing the preprocessing functions
      healthy_files09 (str): Path containing the healthy data rotating at 900 rpm
      healthy_files15 (str): Path containing the healthy data rotating at 1500 rpm
      inner_files09 (str): Path containing the inner ring data rotating at 900 rpm
      inner_files15 (str): Path containing the inner data rotating at 1500 rpm
      outer_files09 (str): Path containing the outer ring data rotating at 900 rpm
      outer_files15 (str): Path containing the outer data rotating at 1500 rpm


    Attributes:
      healthy_files09 (str): Path containing the healthy data rotating at 900 rpm
      healthy_files15 (str): Path containing the healthy data rotating at 1500 rpm
      inner_files09 (str): Path containing the inner ring data rotating at 900 rpm
      inner_files15 (str): Path containing the inner data rotating at 1500 rpm
      outer_files09 (str): Path containing the outer ring data rotating at 900 rpm
      outer_files15 (str): Path containing the outer data rotating at 1500 rpm

    """

    def __init__(self, preprocessing, healthy_files09, healthy_files15, inner_files09, inner_files15, outer_files09,
                 outer_files15):
        self.preprocessing = preprocessing
        self.healthy_files09 = healthy_files09
        self.healthy_files15 = healthy_files15
        self.inner_files09 = inner_files09
        self.inner_files15 = inner_files15
        self.outer_files09 = outer_files09
        self.outer_files15 = outer_files15

    def createDataSets(self, files, defect, load, length=1000, noise = True, amplitude_noise = 0.05):
        """
        Create processed Paderborn datasets. Uses the preprocessing functions

        Args:
          files (list of strings): Names of the files to load the data from
          defect (int): Type of failure of the bearing (indexed: 0->healthy, 1-> Inner ring, 2-> Outer ring)
          load (int): Type of load (rotation speed) (0-> 900 rpm, 1-> 1500 rpm)
          length (int): Input length for the model

        Returns:
          X_train (numpy array): Processed X train data
          y_train (numpy array): Labels of X_train
          load_train (numpy array): Labels of the velocities/loads for X_train
          X_test (numpy array): Processed X test data
          y_test (numpy array): Labels of X_test
          load_test (numpy array): Labels of the velocities/loads for X_test
          amplitude_noise (float): Amplitude of the added noise
        """
        # Prepare data frames for train and test data
        data_train = pd.DataFrame(np.zeros((1, length)))
        data_train.columns = ["t" + str(i) for i in range(length)]

        data_test = pd.DataFrame(np.zeros((1, length)))
        data_test.columns = ["t" + str(i) for i in range(length)]

        i = 0
        # Load data for each file name
        for file_name in files:
            print(file_name)
            mat_dict = scipy.io.loadmat(file_name)  # Load data
            # You have to do strange things to get the data
            try:
                data = mat_dict[file_name[24:43]]['Y'][0][0][0][6][2][0]
            except:
                data = mat_dict[file_name[24:42]]['Y'][0][0][0][6][2][0]
            data = scipy.signal.decimate(data, 12)  # Downsampling to 5000 Hz
            vel = float(file_name[21:23])

            if noise:
                data = self.preprocessing.add_noise_harmonics(data, vel)  # Adding noise to the first harmonics

            aux = self.preprocessing.splitData(data, defect, load,
                                               length, A_sd=amplitude_noise, noise = noise)  # Split data into the input lenght and add band noise
            if i % 2 == 0:  # Separate half of the data to training and half to test
                data_train = pd.concat([data_train, aux])
            else:
                data_test = pd.concat([data_test, aux])
            i += 1
        # Prepare data frames
        data_train = data_train.drop(0)
        data_test = data_test.drop(0)

        X_train = data_train.values[:, 2:]
        y_train = np.repeat(defect, X_train.shape[0])  #
        load_train = np.repeat(load, X_train.shape[0])

        X_test = data_test.values[:, 2:]
        y_test = np.repeat(defect, X_test.shape[0])
        load_test = np.repeat(load, X_test.shape[0])

        return X_train, X_test, y_train, y_test, load_train, load_test

    def get_X_y(self, length=1000, noise = True, amplitude_noise = 0.05):
        """
        Wrap-up function to generate Paderborn training and test datasets

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

        # Get Healthy datasets
        X_healthy_09_train, X_healthy_09_test, y_healthy_09_train, y_healthy_09_test, load_healthy_09_train, load_healthy_09_test = self.createDataSets(
            self.healthy_files09, 0, 4, length, noise = noise, amplitude_noise = amplitude_noise)
        X_healthy_15_train, X_healthy_15_test, y_healthy_15_train, y_healthy_15_test, load_healthy_15_train, load_healthy_15_test = self.createDataSets(
            self.healthy_files15, 0, 5, length, noise = noise, amplitude_noise = amplitude_noise)

        # Get Inner ring datasets
        X_inner_09_train, X_inner_09_test, y_inner_09_train, y_inner_09_test, load_inner_09_train, load_inner_09_test = self.createDataSets(
            self.inner_files09, 1, 4, length, noise = noise, amplitude_noise = amplitude_noise)
        X_inner_15_train, X_inner_15_test, y_inner_15_train, y_inner_15_test, load_inner_15_train, load_inner_15_test = self.createDataSets(
            self.inner_files15, 1, 5, length, noise = noise, amplitude_noise = amplitude_noise)

        # Get Outer ring datasets
        X_outer_09_train, X_outer_09_test, y_outer_09_train, y_outer_09_test, load_outer_09_train, load_outer_09_test = self.createDataSets(
            self.outer_files09, 2, 4, length, amplitude_noise = amplitude_noise)
        X_outer_15_train, X_outer_15_test, y_outer_15_train, y_outer_15_test, load_outer_15_train, load_outer_15_test = self.createDataSets(
            self.outer_files15, 2, 5, length, noise = noise, amplitude_noise = amplitude_noise)

        # Concatenate all  datsets
        X_test_Pad = np.concatenate(
            (X_healthy_09_test, X_inner_09_test, X_outer_09_test, X_healthy_15_test, X_inner_15_test, X_outer_15_test),
            axis=0)
        X_train_Pad = np.concatenate((X_healthy_09_train, X_inner_09_train, X_outer_09_train, X_healthy_15_train,
                                      X_inner_15_train, X_outer_15_train), axis=0)

        # Concatenate all labels datasets
        y_test_Pad = np.concatenate(
            (y_healthy_09_test, y_inner_09_test, y_outer_09_test, y_healthy_15_test, y_inner_15_test, y_outer_15_test),
            axis=0)
        y_train_Pad = np.concatenate((y_healthy_09_train, y_inner_09_train, y_outer_09_train, y_healthy_15_train,
                                      y_inner_15_train, y_outer_15_train), axis=0)

        # Concatenate load names
        load_test_Pad = np.concatenate((load_healthy_09_test, load_inner_09_test, load_outer_09_test,
                                        load_healthy_15_test, load_inner_15_test, load_outer_15_test), axis=0)
        load_train_Pad = np.concatenate((load_healthy_09_train, load_inner_09_train, load_outer_09_train,
                                         load_healthy_15_train, load_inner_15_train, load_outer_15_train), axis=0)
        return X_train_Pad, X_test_Pad, y_train_Pad, y_test_Pad, load_train_Pad, load_test_Pad

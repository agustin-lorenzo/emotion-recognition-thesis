import fcwt
import numpy as np
import pickle
import math
import copy
import random
import h5py

# Constants
SNR = 5                # signal-to-noise ratio
REP_FACTOR = 5         # total samples per original (1 original + REP_FACTOR-1 augmented copies)
WINDOW_SIZE = 5 * 128  # 5-second samples at 128 Hz (640 frames per sample)
TRAIN_PROB = 0.8       # train/test split probability

# helper function for adding gaussian noise to a frame
def add_gaussian_noise(signal, snr_db=5):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, math.sqrt(noise_power), signal.shape)
    return signal + noise

# creating h5 files to avoid keeping all data in memory at once
train_file = h5py.File("data/deap_train_data.h5", "w")
test_file = h5py.File("data/deap_test_data.h5", "w")

train_samples = train_file.create_dataset(
    "samples", shape=(0, WINDOW_SIZE, 32, 32),
    maxshape=(None, WINDOW_SIZE, 32, 32), dtype=np.float32, chunks=True
)
train_labels = train_file.create_dataset(
    "labels", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=True
)

test_samples = test_file.create_dataset(
    "samples", shape=(0, WINDOW_SIZE, 32, 32),
    maxshape=(None, WINDOW_SIZE, 32, 32), dtype=np.float32, chunks=True
)
test_labels = test_file.create_dataset(
    "labels", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=True
)

train_count = 0
test_count = 0

# 32 participants in dataset
for subject in range(32):
    file_path = f"data_preprocessed_python/s{subject + 1:02}.dat"
    x = pickle.load(open(file_path, 'rb'), encoding='latin1')
    data = x['data']
    labels = x['labels']
    relevant_channels = data[:, :32, :]
    relevant_labels = labels[:, :2]
    classes = []
    
    # assign valence-arousal classes for all trials
    for trial in range(40):
        valence, arousal = relevant_labels[trial][0], relevant_labels[trial][1]
        if valence < 4.5 and arousal < 4.5:
            cls = 0
        elif valence < 4.5:
            cls = 1
        elif arousal < 4.5:
            cls = 2
        else:
            cls = 3
        classes.append(cls)
    
    # CWT parameters
    fs = 128 # hz
    f0 = 4 # lowest frequency
    f1 = 45 # highest frequency
    fn = 32  # number of frequencies (matches 32 channels to make square frame)
    
    # 40 trials per participant
    for trial in range(40):
        total_cwt = np.zeros((32 * fn, 8064))
        # get single image with all CWT data over entire trial
        for channel in range(32):
            signal = relevant_channels[trial][channel]
            _, current_cwt = fcwt.cwt(signal, fs, f0, f1, fn)
            start = channel * fn
            end = (channel + 1) * fn
            total_cwt[start:end, :] = np.abs(current_cwt) ** 2 # square coefficients to get energy
        
        # reshape 2D image into frames that make up 3D video
        all_frames = []
        for sample in range(8064):
            frame = np.zeros((32, 32), dtype=np.float32)
            for channel in range(32):
                start = channel * fn
                end = (channel + 1) * fn
                frame[:, channel] = total_cwt[start:end, sample]
            all_frames.append(frame)
        
        # split into window_size hz clips
        for i in range(0, 8064, WINDOW_SIZE):
            if i + WINDOW_SIZE > 8064:
                break
            sample_segment = np.array(all_frames[i:i+WINDOW_SIZE], dtype=np.float32)
            # # standardize original sample
            # sample_mean = sample_segment.mean()
            # sample_std = sample_segment.std()
            # if sample_std == 0:
            #     sample_std = 1
            # sample_segment = (sample_segment - sample_mean) / sample_std
            # # normalize original sample
            # sample_min = sample_segment.min()
            # sample_max = sample_segment.max()
            # sample_segment = (sample_segment - sample_min) / (sample_max - sample_min)
            
            # new normalization: follows Li et al., all values sum to 1, appropriate for ANNs
            total_energy = np.sum(sample_segment)
            if total_energy == 0:
                total_energy = 1  # prevent division by zero
            sample_segment = sample_segment / total_energy
            
            # randomly decide if the sample goes to the training set with train_prob% chance
            if random.random() < TRAIN_PROB:
                # add original sample to training set
                train_samples.resize(train_count + 1, axis=0)
                train_samples[train_count] = sample_segment
                train_labels.resize(train_count + 1, axis=0)
                train_labels[train_count] = int(classes[trial])
                train_count += 1
                
                # augment the original sample rep_factor-1 times
                for _ in range(REP_FACTOR - 1):
                    augmented_sample = []
                    for frame in sample_segment:
                        noisy_frame = add_gaussian_noise(frame, SNR)
                        augmented_sample.append(noisy_frame)
                    augmented_sample = np.array(augmented_sample, dtype=np.float32)
                    # restandardize augmented sample
                    sample_mean = augmented_sample.mean()
                    sample_std = augmented_sample.std()
                    if sample_std == 0:
                        sample_std = 1
                    augmented_sample = (augmented_sample - sample_mean) / sample_std
                    # renormalize augmented sample
                    sample_min = augmented_sample.min()
                    sample_max = augmented_sample.max()
                    augmented_sample = (augmented_sample - sample_min) / (sample_max - sample_min)
                    # add the augmented sample
                    train_samples.resize(train_count + 1, axis=0)
                    train_samples[train_count] = augmented_sample
                    train_labels.resize(train_count + 1, axis=0)
                    train_labels[train_count] = int(classes[trial])
                    train_count += 1
            else:
                # add original sample to test set without augmenting
                test_samples.resize(test_count + 1, axis=0)
                test_samples[test_count] = sample_segment
                test_labels.resize(test_count + 1, axis=0)
                test_labels[test_count] = int(classes[trial])
                test_count += 1
    
    print(f"Processed subject {subject+1}: Train samples so far: {train_count}, Test samples so far: {test_count}", end='\r')

train_file.close()
test_file.close()
print("\nData saved to 'data/deap_train_data.h5' and 'data/deap_test_data.h5'.")

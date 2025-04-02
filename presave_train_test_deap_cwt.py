import fcwt
import numpy as np
import pickle
import math
import copy
import random
import h5py

# Constants
SNR = 5                # Signal-to-noise ratio for Gaussian noise
REP_FACTOR = 5         # Total samples per original (1 original + REP_FACTOR-1 augmented copies)
WINDOW_SIZE = 5 * 128  # 5-second samples at 128 Hz (640 frames per sample)
TRAIN_PROB = 0.8       # Approximate probability a sample is assigned to training

# Helper: Add Gaussian noise
def add_gaussian_noise(signal, snr_db=5):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, math.sqrt(noise_power), signal.shape)
    return signal + noise

# Create HDF5 files for training and testing with extendable datasets
train_file = h5py.File("deap_train_data.h5", "w")
test_file = h5py.File("deap_test_data.h5", "w")

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

# Process each subject
for subject in range(32):
    file_path = f"data_preprocessed_python/s{subject + 1:02}.dat"
    x = pickle.load(open(file_path, 'rb'), encoding='latin1')
    data = x['data']
    labels = x['labels']
    relevant_channels = data[:, :32, :]
    relevant_labels = labels[:, :2]
    classes = []
    
    # assign valence-arousal class
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
    fs = 128
    f0 = 4
    f1 = 45
    fn = 32  # number of frequencies (matches the 32-channel square frame)
    
    for trial in range(40):
        total_cwt = np.zeros((32 * fn, 8064))
        for channel in range(32):
            signal = relevant_channels[trial][channel]
            _, current_cwt = fcwt.cwt(signal, fs, f0, f1, fn)
            start = channel * fn
            end = (channel + 1) * fn
            total_cwt[start:end, :] = np.abs(current_cwt)
        
        # Convert the 2D CWT result into a series of frames (each frame: 32x32)
        all_frames = []
        for sample in range(8064):
            frame = np.zeros((32, 32), dtype=np.float32)
            for channel in range(32):
                start = channel * fn
                end = (channel + 1) * fn
                frame[:, channel] = total_cwt[start:end, sample]
            all_frames.append(frame)
        
        # Split the trial into segments of WINDOW_SIZE frames
        for i in range(0, 8064, WINDOW_SIZE):
            if i + WINDOW_SIZE > 8064:
                break
            sample_segment = np.array(all_frames[i:i+WINDOW_SIZE], dtype=np.float32)
            
            # Decide randomly whether to assign the sample to training or testing
            if random.random() < TRAIN_PROB:
                # Append original sample to training dataset
                train_samples.resize(train_count + 1, axis=0)
                train_samples[train_count] = sample_segment
                train_labels.resize(train_count + 1, axis=0)
                train_labels[train_count] = int(classes[trial])
                train_count += 1
                
                # Augment: add REP_FACTOR-1 copies (augmented only for training)
                for _ in range(REP_FACTOR - 1):
                    augmented_sample = []
                    for frame in sample_segment:
                        noisy_frame = add_gaussian_noise(frame, SNR)
                        augmented_sample.append(noisy_frame)
                    augmented_sample = np.array(augmented_sample, dtype=np.float32)
                    train_samples.resize(train_count + 1, axis=0)
                    train_samples[train_count] = augmented_sample
                    train_labels.resize(train_count + 1, axis=0)
                    train_labels[train_count] = int(classes[trial])
                    train_count += 1
            else:
                # Append sample to testing dataset (no augmentation)
                test_samples.resize(test_count + 1, axis=0)
                test_samples[test_count] = sample_segment
                test_labels.resize(test_count + 1, axis=0)
                test_labels[test_count] = int(classes[trial])
                test_count += 1
    
    print(f"Processed subject {subject+1}: Train samples so far: {train_count}, Test samples so far: {test_count}")

train_file.close()
test_file.close()
print("Data saved to 'train_data.h5' and 'test_data.h5'.")

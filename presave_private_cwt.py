import pandas as pd
import numpy as np
import fcwt
import h5py
import math
import random

# initialize constant variables
SNR = 5                # signal-to-noise ratio
REP_FACTOR = 10        # total samples per original (1 original + REP_FACTOR-1 augmented copies)
WINDOW_SIZE = 64       # 2-second samples at 128 Hz -> 256 frames -> averaged to 64 frames
TRAIN_PROB = 0.8       # train/test split probability

# paramaters for calculating cwt, not to be changed
fs = 128 
f0 = 4 # lowest frequency
f1 = 45 # highest frequency
fn = 128 # number of frequencies, match channel number for square frame

# helper function for adding gaussian noise to a frame
def add_gaussian_noise(signal, snr_db=5):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, math.sqrt(noise_power), signal.shape)
    return signal + noise

# helper function used for adding a sample to train/test .h5 files
def add_to_h5(sample_list, label_list, count, sample, cls):
    sample_list.resize(count + 1, axis=0)
    sample_list[count] = sample
    label_list.resize(count + 1, axis=0)
    label_list[count] = np.uint8(cls)

# helper function for normalizing a sample, sample has to be list of frames
def normalize_sample(all_frames):
    sample = np.array(all_frames, dtype=np.float32)
    total_energy = np.sum(sample)
    if total_energy == 0:
        total_energy = 1
    sample = sample / total_energy
    return sample

# helper function for getting sample video from 128 channel .csv rows for one trial
def get_sample(trial_rows, augmentation=False):
    total_cwt = np.zeros((128 * fn, 256)) # image containing cwt values for all channels, change width depending on video/picture data
    # iterate through channels to get 2D CWT image
    for channel, (_, row) in enumerate(trial_rows.iterrows()):
        signal_cols = [str(i) for i in range(1, 257)] # change width depending on wideo/piture data
        signal = row[signal_cols].to_numpy()
        if augmentation:
            signal = add_gaussian_noise(signal)
        _, current_cwt = fcwt.cwt(signal, fs, f0, f1, fn)
        start = channel * fn
        end = (channel + 1) * fn
        total_cwt[start:end, :] = abs(current_cwt) ** 2 # square coefficients to get energy
    
    # reshape so that time dimension is split into segments and average over each segment
    # averages cwt data from 256 frames down to 64 frames
    num_frames = 256 // 4
    total_cwt = total_cwt.reshape(128 * fn, num_frames, 4).mean(axis=2)    
        
    # create list of frames
    all_frames = []
    for time_point in range(num_frames):
        frame = np.zeros((128, 128))
        for channel in range(128):
            start = channel * fn
            end = (channel + 1) * fn
            frame[:, channel] = total_cwt[start:end, time_point]
        all_frames.append(frame) # append 1-channel frame, no rgb
    # normalize and return sample
    sample = normalize_sample(all_frames)
    return sample

print("opening dataset...")
df = pd.read_csv('data/preprocessed_picture.csv')
print("dataset opened.")

# creating h5 files to avoid keeping all data in memory at once
train_file = h5py.File("data/private_train_data.h5", "w")
test_file = h5py.File("data/private_test_data.h5", "w")

train_samples = train_file.create_dataset(
    "samples", shape=(0, WINDOW_SIZE, 128, 128),
    maxshape=(None, WINDOW_SIZE, 128, 128), dtype=np.float32, chunks=True
)
train_labels = train_file.create_dataset(
    "labels", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=True
)

test_samples = test_file.create_dataset(
    "samples", shape=(0, WINDOW_SIZE, 128, 128),
    maxshape=(None, WINDOW_SIZE, 128, 128), dtype=np.float32, chunks=True
)
test_labels = test_file.create_dataset(
    "labels", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=True
)

train_count = 0
test_count = 0

# iterate through participants
for participant in df['par_id'].unique():
    participant_rows = df[df['par_id'] == participant]
    
    # iterate through trials
    for stimulus in participant_rows['Stim_name'].unique():
        # TRAIN_PROB% chance that given trial goes to training set
        if random.random() < TRAIN_PROB: # add original + augmented samples to training set
            trial_rows = participant_rows[participant_rows['Stim_name'] == stimulus]
            cls = np.uint8(trial_rows.iloc[0]["class"])
            sample = get_sample(trial_rows)
            add_to_h5(train_samples, train_labels, train_count, sample, cls)
            train_count += 1
            print(f"\tTrain samples so far: {train_count}, Test samples so far: {test_count}", end='\r')   
                
            for _ in range(REP_FACTOR - 1): # augment each channel REP_FACTOR number of times
                #  iterate through channels again to get augmented sample
                augmented_sample = get_sample(trial_rows, augmentation=True)
                add_to_h5(train_samples, train_labels, train_count, augmented_sample, cls)
                train_count += 1
                print(f"\tTrain samples so far: {train_count}, Test samples so far: {test_count}", end='\r')   
                
        else: # add to test set without augmentation
            trial_rows = participant_rows[participant_rows['Stim_name'] == stimulus]
            cls = np.uint8(trial_rows.iloc[0]["class"])
            sample = get_sample(trial_rows)
            add_to_h5(test_samples, test_labels, test_count, sample, cls)
            test_count += 1
            print(f"\tTrain samples so far: {train_count}, Test samples so far: {test_count}", end='\r')
        
    print(f"\nProcessed subject {participant}.")

train_file.close()
test_file.close()
print("\nData saved to 'data/private_train_data.h5' and 'data/private_test_data.h5'.")

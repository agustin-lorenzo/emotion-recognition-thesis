import pandas as pd
import numpy as np
import fcwt
import h5py
import math
import random

# initialize constant variables
SNR = 5                # signal-to-noise ratio
REP_FACTOR = 10         # total samples per original (1 original + REP_FACTOR-1 augmented copies)
WINDOW_SIZE = 2 * 128  # 5-second samples at 128 Hz (640 frames per sample)
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
    label_list[count] = int(cls)

# helper function for normalizing a sample, sample has to be list of frames
def normalize_sample(all_frames):
    sample = np.array(all_frames, dtype=np.float32)
    total_energy = np.sum(sample)
    if total_energy == 0:
        total_energy = 1
    sample = sample / total_energy
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

#trials = [] # list of cwt videos for all trials
#labels = [] # classes corresponding to all trials

train_count = 0
test_count = 0

# iterate through participants
for participant in df['par_id'].unique():
    participant_rows = df[df['par_id'] == participant]
    
    # iterate through trials
    for stimulus in participant_rows['Stim_name'].unique():
        trial_rows = participant_rows[participant_rows['Stim_name'] == stimulus]
        cls = trial_rows.iloc[0]["class"]
        
        total_cwt = np.zeros((128 * fn, 256)) # image containing cwt values for all channels, change width depending on video/picture data
        # iterate through channels
        for channel, (_, row) in enumerate(trial_rows.iterrows()):
            signal_cols = [str(i) for i in range(1, 257)] # change width depending on wideo/piture data
            signal = row[signal_cols].to_numpy()
            _, current_cwt = fcwt.cwt(signal, fs, f0, f1, fn)
            start = channel * fn
            end = (channel + 1) * fn
            total_cwt[start:end, :] = abs(current_cwt) ** 2 # square coefficients to get energy
        
        # convert 2D time sample x channel-frequency format (x, y) to 3D channel x frequency x time format (x, y, z)
        # each 'frame' in all_frames is a 2D image showing each channel's CWT value for that time sample
        all_frames = []
        for sample in range(256):
            frame = np.zeros((128, 128))
            for channel in range(128):
                start = channel * fn
                end = (channel + 1) * fn
                frame[:, channel] = total_cwt[start:end, sample]
            
            #norm_frame = (frame - frame.min()) / (frame.max() - frame.min()) # normalize frame
            #scaled_frame = (norm_frame * 255).astype(np.uint8) # scale to 255 for video
            all_frames.append(frame) # append 1-channel frame, no rgb
            
        # split trial into x-second samples as suggested in Arjun et al. 2021, only necessary for video data
        # for i in range(0, 1280, WINDOW_SIZE): 
        #     if i + WINDOW_SIZE > 1280:
        #         break
        #     clip = all_frames[i:i+WINDOW_SIZE]
        #     trials.append(np.array(clip, dtype=np.uint8)) # append on this level for clipping
        #     labels.append(int(cls))
        
        # normalize sample, ensure total energy sums to 1
        sample = normalize_sample(all_frames)
        
        # randomly decide if the sample goes to the training set with train_prob% chance
        if random.random() < TRAIN_PROB:
            add_to_h5(train_samples, train_labels, train_count, sample, cls)
            # train_samples.resize(train_count + 1, axis=0)
            # train_samples[train_count] = sample
            # train_labels.resize(train_count + 1, axis=0)
            # train_labels[train_count] = int(cls)
            
            for _ in range(REP_FACTOR - 1):
                all_augmented_frames = []
                for frame in sample:
                    noisy_frame = add_gaussian_noise(frame, SNR)
                    all_augmented_frames.append(noisy_frame)
                # redo normalization
                augmented_sample = normalize_sample(all_augmented_frames)
                # add the augmented sample
                add_to_h5(train_samples, train_labels, train_count, augmented_sample, cls)
                # train_samples.resize(train_count + 1, axis=0)
                # train_samples[train_count] = augmented_sample
                # train_labels.resize(train_count + 1, axis=0)
                # train_labels[train_count] = int(cls)
        else: # add to test set without augmentation
            add_to_h5(test_samples, test_labels, test_count, sample, cls)
        
    print(f"Processed subject {participant}: Train samples so far: {train_count}, Test samples so far: {test_count}", end='\r')   
        # print(f"\tCurrent participant: {participant}\tCurrent Number of Trials: {len(trials)}", end='\r')
        # trials.append(np.array(all_frames, dtype=np.float32)) # append on this level for no clipping
        # labels.append(int(cls))
train_file.close()
test_file.close()
print("\nData saved to 'data/private_train_data.h5' and 'data/private_test_data.h5'.")
# trials = np.array(trials, dtype=np.uint8)
# labels = np.array(labels, dtype=np.uint8)
# global_min = trials.min()
# global_max = trials.max()
# trials = (trials - global_min) / (global_max - global_min) # normalize all trials to [0, 1]

        
# print("\nsaving features...")
# trials_array = np.stack(trials)  # shape: (num_trials, 128, 128, 128 * seconds)
# del trials
# gc.collect()

# labels_array = np.array(labels)
# del labels
# gc.collect()

# print("\tTrials shape:", trials.shape)  # should be (num_trials, 128, 128, 128 * seconds)
# print("\tTrials dtype:", trials.dtype)

# print("\n\tLabels shape:", labels.shape)  # should be (num_trials,)
# print("\tLabels dtype:", labels.dtype)
# np.savez_compressed('data/extracted_features_video.npz', trials=trials, labels=labels)
# print("done.")
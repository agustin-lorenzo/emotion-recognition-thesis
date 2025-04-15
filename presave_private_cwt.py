import fcwt
import numpy as np
import pandas as pd
import pickle
import math
from run_k_folds import add_gaussian_noise

# initialize constants
SNR = 5  # signal-to-noise ratio
REP_FACTOR = 0  # number of augmented samples per original sample
WINDOW_SIZE = 32  # 640 frames per sample
NUM_FRAMES = 8064 // 4
SEG_LENGTH = 64        # We expect 64 frames per 2-second segment after averaging
DESIRED_FRAMES = 32    # We want each output clip to have 32 frames
STRIDE = 2

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

# helper function for normalizing a sample, sample has to be list of frames
def normalize_sample(all_frames):
    sample = np.array(all_frames, dtype=np.float32)
    total_energy = np.sum(sample)
    if total_energy == 0:
        total_energy = 1
    sample = sample / total_energy
    return sample

# helper function for getting sample video from 128 channel .csv rows for one trial
def get_sample(trial_rows, seconds=2, augmentation=False, normalize=False):
    time_points = 128 * seconds
    total_cwt = np.zeros((128 * fn, time_points)) # image containing cwt values for all channels, change width depending on video/picture data
    # iterate through channels to get 2D CWT image
    for channel, (_, row) in enumerate(trial_rows.iterrows()):
        signal_cols = [str(i) for i in range(1, time_points + 1)] # change width depending on wideo/piture data
        signal = row[signal_cols].to_numpy()
        if augmentation:
            signal = add_gaussian_noise(signal)
        _, current_cwt = fcwt.cwt(signal, fs, f0, f1, fn)
        start = channel * fn
        end = (channel + 1) * fn
        total_cwt[start:end, :] = abs(current_cwt) # ** 2 # square coefficients to get energy
    
    # reshape so that time dimension is split into segments and average over each segment
    # averages cwt data from 256 frames down to 64 frames
    num_frames = time_points // 4
    total_cwt = total_cwt.reshape(128 * fn, num_frames, 4).mean(axis=2)    
        
    # create list of frames
    all_frames = []
    for t in range(num_frames):
        frame = np.zeros((128, 128))
        for channel in range(128):
            start = channel * fn
            end = (channel + 1) * fn
            frame[:, channel] = total_cwt[start:end, t]
        all_frames.append(frame) # append 1-channel frame, no rgb
    # use a stride of 2 to reduce frames down to 32, expected by vivit
    clip = all_frames[::2]
    # normalize and return sample
    if normalize: 
        return normalize_sample(clip)
    else:
        return np.array(clip, dtype=np.float32)

all_samples = []
all_labels = []

print("opening picture dataset...")
df = pd.read_csv('data/preprocessed_picture.csv')
print("picture dataset opened.")

for participant in df['par_id'].unique():
    participant_rows = df[df['par_id'] == participant]
    
    # iterate through trials
    for stimulus in participant_rows['Stim_name'].unique():
        trial_rows = participant_rows[participant_rows['Stim_name'] == stimulus]
        cls = np.uint8(trial_rows.iloc[0]["class"])
        all_samples.extend(get_sample(trial_rows))
        all_labels.extend(cls)

print("picture data processed.")
print("\nopening video dataset...")
df = pd.read_csv('data/preprocessed_video.csv')
print("video dataset opened.")

for participant in df['par_id'].unique():
    participant_rows = df[df['par_id'] == participant]
    
    # iterate through trials
    for stimulus in participant_rows['Stim_name'].unique():
        trial_rows = participant_rows[participant_rows['Stim_name'] == stimulus]
        cls = np.uint8(trial_rows.iloc[0]["class"])
        sample = get_sample(trial_rows, seconds=10)
        clip_length = 32 # expected by vivit
        for i in range(5):
            start_idx = i * clip_length
            end_idx = (i + 1) * clip_length
            clip = sample[start_idx:end_idx] # what do I do here
            all_samples.append(clip)
            all_labels.append(cls)

np.savez("all_private_cwt_data.npz", samples=np.array(all_samples, dtype=np.float32), labels=np.array(all_labels, dtype=np.uint8))
print("\nData saved to all_private_cwt_data.npz")

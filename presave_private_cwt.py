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
def get_sample(trial_rows, augmentation=False, normalize=False):
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
    if normalize: 
        return normalize_sample(all_frames)
    else:
        return np.array(all_frames, dtype=np.float32)

all_samples = []
all_labels = []

print("opening dataset...")
df = pd.read_csv('data/preprocessed_picture.csv')
print("dataset opened.")

for participant in df['par_id'].unique(): # TODO: finish updating script for new vivit finetuning
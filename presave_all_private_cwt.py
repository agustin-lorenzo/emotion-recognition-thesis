import fcwt
import numpy as np
import pandas as pd
import math
from old_scripts.run_k_folds import add_gaussian_noise

# initialize constants
SNR = 5  # signal-to-noise ratio
REP_FACTOR = 0  # number of augmented samples per original sample
WINDOW_SIZE = 32  # 640 frames per sample
NUM_FRAMES = 8064 // 4
SEG_LENGTH = 64        
DESIRED_FRAMES = 32    
STRIDE = 2

# get private dataset's original channel order from first sample in preprocessed picture csv
df_pic = pd.read_csv("data/preprocessed_picture.csv")
first_pid  = df_pic["par_id"].iloc[0]
first_stim = df_pic[df_pic["par_id"] == first_pid]["Stim_name"].iloc[0]
trial0     = df_pic[
    (df_pic["par_id"]   == first_pid) &
    (df_pic["Stim_name"] == first_stim)
]
private_order = trial0["channel"].tolist()
print("private order", private_order)
print("private order length: ", len(private_order))
hilbert_order =['PO10', 'POO10H', 'I2', 'OI2H', 'O2', 'OZ', 'POO2', 'PO4', 'PO8', 'PPO6H', 'P6', 'P4', 'PPO2H', 'P2', 'CPP2H', 'CPZ', 'CP2', 'CP4', 'CPP4H', 'CPP6H', 'CP6', 'TP8', 'TPP8H', 'P8', 'P10', 'PPO10H', 'TP10', 'FTT10H', 'TTP8H', 'FTT8H', 'T8', 'FT8', 'FT10', 'FFT10H', 'F10', 'AF8', 'AFF6H', 'F8', 'FFT8H', 'FFC6H', 'FFC4H', 'F4', 'F6', 'AF4', 'AFP2', 'AFF2H', 'FP2', 'FPZ', 'AFZ', 'F2', 'FFC2H', 'CZ', 'C2', 'FC2', 'FC4', 'FC6', 'C6', 'C4', 'FCC4H', 'FCC6H', 'CCP6H', 'CCP4H', 'CCP2H', 'FCC2H', 'FCC1H', 'CCP1H', 'CCP3H', 'CCP5H', 'FCC5H', 'FCC3H', 'C3', 'C5', 'FC5', 'FC3', 'FC1', 'C1', 'FCZ', 'FFC1H', 'F1', 'FZ', 'FP1', 'AFF1H', 'AFP1', 'AF3', 'F5', 'F3', 'FFC3H', 'FFC5H', 'FFT7H', 'F7', 'AFF5H', 'AF7', 'F9', 'FFT9H', 'FT9', 'FT7', 'T7', 'FTT7H', 'TTP7H', 'FTT9H', 'TP9', 'PPO9H', 'P9', 'P7', 'TPP7H', 'TP7', 'CP5', 'CPP5H', 'CPP3H', 'CP3', 'CP1', 'CPP1H', 'P1', 'PZ', 'PPO1H', 'P3', 'P5', 'PPO5H', 'PO7', 'PO3', 'POO1', 'POZ', 'O1', 'OI1H', 'IZ', 'I1', 'POO9H', 'PO9']
print("hilbert order length: ", len(hilbert_order))
private_set = set(private_order)
hilbert_set = set(hilbert_order)

missing_from_hilbert = private_set   - hilbert_set
extra_in_hilbert   = hilbert_set     - private_set

print(f"Channels in data but not in your Hilbert list ({len(missing_from_hilbert)}):")
for ch in sorted(missing_from_hilbert):
    print("  ", ch)

print(f"\nChannels in your Hilbert list but not in data ({len(extra_in_hilbert)}):")
for ch in sorted(extra_in_hilbert):
    print("  ", ch)


channel_order = [private_order.index(ch) for ch in hilbert_order]

# paramaters for calculating cwt, not to be changed
fs = 128 # signal hz
f0 = 4   # lowest frequency
f1 = 45  # highest frequency
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
    trial_rows = trial_rows.iloc[channel_order]
    time_points = 128 * seconds
    total_cwt = np.zeros((128 * fn, time_points)) # image containing cwt values for all channels, change width depending on video/picture data
    # iterate through channels to get 2D CWT image
    for channel, (_, row) in enumerate(trial_rows.iterrows()): #TODO: change order of channels to match order obtained by Hilbert curve
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

# processing picture dataset
print("opening picture dataset...")
df = pd.read_csv('data/preprocessed_picture.csv')
print("picture dataset opened.")

for participant in df['par_id'].unique():
    participant_rows = df[df['par_id'] == participant]
    
    # iterate through trials
    for stimulus in participant_rows['Stim_name'].unique():
        trial_rows = participant_rows[participant_rows['Stim_name'] == stimulus]
        cls = np.uint8(trial_rows.iloc[0]["class"])
        all_samples.append(get_sample(trial_rows))
        all_labels.append(cls)
print("picture data processed.")

# processing video dataset for 2s clips
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
            clip = sample[start_idx:end_idx]
            all_samples.append(clip)
            all_labels.append(cls)
print("video dataset processed.")

np.savez("all_private_cwt_data.npz", samples=np.array(all_samples, dtype=np.float32), labels=np.array(all_labels, dtype=np.uint8))
print("\nData saved to all_private_cwt_data.npz")

import fcwt
import numpy as np
import pickle
import math
import copy
from sklearn.model_selection import train_test_split

# initialize constants
SNR = 5 # signal-to-noise ratio
REP_FACTOR = 4 # how many augmented samples are created out of one original sample

# helper method for adding gaussian noise to each frame in a sample, includes signal-to-noise parameter
def add_gaussian_noise(signal, snr_db=5):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, math.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

six_sec_samples = [] # all 6-second samples from all subjects in the dataset
sample_classes = []  # all target classes for each of the 6-second samples

for subject in range(32):
    file_path = f"data_preprocessed_python/s{subject + 1:02}.dat"
    x = pickle.load(open(file_path, 'rb'), encoding='latin1')
    data = x['data']
    labels = x['labels']
    relevant_channels = data[:, :32, :]
    relevant_labels = labels[:, :2]
    classes = []

    for trial in range(40):
        # 4 valence-arousal classes
        valence, arousal = relevant_labels[trial][0], relevant_labels[trial][1]
        if valence < 4.5 and arousal < 4.5: # LVLA
            cls = 0
        elif valence < 4.5:                 # LVHA
            cls = 1
        elif arousal < 4.5:                 # HVLA
            cls = 2
        else:                               # HVHA
            cls = 3
        classes.append(cls)

    # initialize constant variables
    # paramaters for calculating cwt, not to be changed
    fs = 128 
    f0 = 4 # lowest frequency
    f1 = 45 # highest frequency
    fn = 32 # number of frequencies, match channel number for square frame

    print(f"Creating frames from participant {subject + 1}'s EEG data...")
    for trial in range(40):
        total_cwt = np.zeros((1024, 8064))

        for channel in range(32):    
            signal = relevant_channels[trial][channel]
            _, current_cwt = fcwt.cwt(signal, fs, f0, f1, fn)
            start = channel * fn
            end = (channel + 1) * fn
            total_cwt[start:end, :] = abs(current_cwt)
        
        # convert 2D time sample x channel-frequency format (x, y) to 3D channel x frequency x time format (x, y, z)
        # each 'frame' in all_frames is a 2D image showing each channel's CWT value for that time sample
        all_frames = []
        for sample in range(8064):
            frame = np.zeros((32, 32))
            for channel in range(32):
                start = channel * fn
                end = (channel + 1) * fn
                frame[:, channel] = total_cwt[start:end, sample]
            # normalize frame
            norm_frame = (frame - frame.min()) / (frame.max() - frame.min())
            # scale to 0-255 (although this is normalized in dataset object later, 
            #                 it's still best to keep the input in a "video format")
            scaled_frame = (norm_frame * 255).astype(np.uint8)
            # stack for pseudo RGB
            frame_rgb = np.stack((scaled_frame,) * 3, axis=0)
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
            all_frames.append(frame_rgb)
            #all_frames.append(frame) # append 1-channel frame without rgb
        
        # split trial into 6-second samples as suggested in Arjun et al. 2021
        window_size = 6 * 128
        for i in range(0, 8064, window_size):
            if i + window_size > 8064:
                break
            original_sample = all_frames[i:i+window_size]
            six_sec_samples.append(original_sample)
            sample_classes.append(classes[trial])
            
            # create 4 augmented copies from original
            # frame as suggested in Li et al. 2016
            for _ in range(REP_FACTOR-1): # commenting out to try on-the-fly augmentation
                augmented_sample = []
                for frame in original_sample:
                    noisy_frame = copy.deepcopy(frame)
                    for channel in range(32):
                        noisy_frame[:, :, channel] = add_gaussian_noise(noisy_frame[:, :, channel], SNR)
                    augmented_sample.append(noisy_frame)
                
                six_sec_samples.append(augmented_sample)
                sample_classes.append(classes[trial])


# save train/test splits to disk
print("All frames initialized.")
print("\nCreating train/test splits...")    
train_data, test_data, train_labels, test_labels = train_test_split(
    six_sec_samples, sample_classes, test_size=0.2, random_state=23, stratify=sample_classes
)
print("Saving to disk...")
np.savez("train_data.npz", samples=train_data, labels=train_labels)
np.savez("test_data.npz", samples=test_data, labels=test_labels)
del six_sec_samples, sample_classes, train_data, test_data, train_labels, test_labels
print("\nDone.")
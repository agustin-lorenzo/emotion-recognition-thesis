import fcwt
import numpy as np
import pickle
import math
import copy
import random
import h5py
from run_k_folds import normalize_sample
from run_k_folds import add_gaussian_noise

# Initialize constants
SNR = 5  # signal-to-noise ratio
REP_FACTOR = 0  # number of augmented samples per original sample
WINDOW_SIZE = 2 * 128  # 640 frames per sample
NUM_FRAMES = 8064 // 4
# Helper method for adding Gaussian noise
def add_gaussian_noise(signal, snr_db=5):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, math.sqrt(noise_power), signal.shape)
    return signal + noise

# Create HDF5 file and datasets with extensible dimensions
with h5py.File("all_deap_cwt_data.h5", "w") as hf:
    samples_dset = hf.create_dataset(
        "samples", shape=(0, WINDOW_SIZE, 32, 32),
        maxshape=(None, WINDOW_SIZE, 32, 32), dtype=np.float32, chunks=True
    )
    labels_dset = hf.create_dataset(
        "labels", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=True
    )
    
    total_samples = 0  # counter for total samples added
    
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
        # for trial in range(40):
        #     valence, arousal = relevant_labels[trial][0], relevant_labels[trial][1]
        #     if valence < 4.5 and arousal < 4.5:  # LVLA
        #         cls = 0
        #     elif valence < 4.5:                 # LVHA
        #         cls = 1
        #     elif arousal < 4.5:                 # HVLA
        #         cls = 2
        #     else:                               # HVHA
        #         cls = 3
        #     classes.append(cls)
        # new labels, 3 classes based on valence alone: unpleasant, neutral, pleasant
        for trial in range(40):
            valence = relevant_labels[trial][0]
            if valence < 3:
                cls = 0
            elif valence < 6:
                cls = 1
            elif valence <= 9:
                cls = 2
            classes.append(cls)

        subject_trials = []  # list for this subject's samples
        subject_labels = []  # list for this subject's labels

        # Parameters for calculating the CWT
        fs = 128 
        f0 = 4
        f1 = 45
        fn = 32  # number of frequencies

        for trial in range(40):
            total_cwt = np.zeros((32 * fn, 8064))
            for channel in range(32):    
                signal = relevant_channels[trial][channel]
                _, current_cwt = fcwt.cwt(signal, fs, f0, f1, fn)
                start = channel * fn
                end = (channel + 1) * fn
                #total_cwt[start:end, :] = np.abs(current_cwt)
                total_cwt[start:end, :] = np.abs(current_cwt) ** 2 # square coefficients to get energy
                
            # average cwt values down by a factor of 4
            total_cwt = total_cwt.reshape(128 * fn, NUM_FRAMES, 4).mean(axis=2)
            
            # Convert to a series of frames: each frame is 32x32
            all_frames = []
            for sample in range(NUM_FRAMES):
                frame = np.zeros((32, 32), dtype=np.float32)
                for channel in range(32):
                    start = channel * fn
                    end = (channel + 1) * fn
                    frame[:, channel] = total_cwt[start:end, sample]
                all_frames.append(frame)
            
            # split trial into smaller window_size second segments, or "views" as described in ViViT paper
            for i in range(0, NUM_FRAMES, WINDOW_SIZE):
                if i + WINDOW_SIZE > 8064:
                    break
                original_sample = all_frames[i:i+WINDOW_SIZE]
                #subject_trials.append(np.array(original_sample, dtype=np.float32))
                # normalize by total energy
                subject_trials.append(normalize_sample(original_sample))
                subject_labels.append(int(classes[trial]))
                
                # add augmented samples if needed, set rep_factor to 0 otherwise
                for _ in range(REP_FACTOR - 1):
                    augmented_sample = []
                    for frame in original_sample:
                        noisy_frame = add_gaussian_noise(frame, SNR)
                        augmented_sample.append(noisy_frame)
                    subject_trials.append(np.array(augmented_sample, dtype=np.float32))
                    subject_labels.append(int(classes[trial]))
        
        # Append this subject's data to the HDF5 datasets
        num_new_samples = len(subject_trials)
        # Resize datasets to accommodate new samples
        samples_dset.resize(total_samples + num_new_samples, axis=0)
        labels_dset.resize(total_samples + num_new_samples, axis=0)
        # Write new samples
        samples_dset[total_samples:total_samples + num_new_samples] = subject_trials
        labels_dset[total_samples:total_samples + num_new_samples] = subject_labels
        total_samples += num_new_samples
        
        print(f"Processed subject {subject+1}: total samples so far = {total_samples}", end='\r')

print("Data saved to all_deap_cwt_data.h5")

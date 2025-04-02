import fcwt
import numpy as np
import pickle
import math
import copy
import random
import h5py

# Initialize constants
SNR = 5  # signal-to-noise ratio
REP_FACTOR = 5  # number of augmented samples per original sample
window_size = 5 * 128  # 640 frames per sample

# Helper method for adding Gaussian noise
def add_gaussian_noise(signal, snr_db=5):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, math.sqrt(noise_power), signal.shape)
    return signal + noise

# Create HDF5 file and datasets with extensible dimensions
with h5py.File("all_deap_cwt_data.h5", "w") as hf:
    samples_dset = hf.create_dataset(
        "samples", shape=(0, window_size, 32, 32),
        maxshape=(None, window_size, 32, 32), dtype=np.float32, chunks=True
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
        for trial in range(40):
            valence, arousal = relevant_labels[trial][0], relevant_labels[trial][1]
            if valence < 4.5 and arousal < 4.5:  # LVLA
                cls = 0
            elif valence < 4.5:                 # LVHA
                cls = 1
            elif arousal < 4.5:                 # HVLA
                cls = 2
            else:                               # HVHA
                cls = 3
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
                total_cwt[start:end, :] = np.abs(current_cwt)
            
            # Convert to a series of frames: each frame is 32x32
            all_frames = []
            for sample in range(8064):
                frame = np.zeros((32, 32), dtype=np.float32)
                for channel in range(32):
                    start = channel * fn
                    end = (channel + 1) * fn
                    frame[:, channel] = total_cwt[start:end, sample]
                all_frames.append(frame)
            
            # Split trial into segments and augment
            for i in range(0, 8064, window_size):
                if i + window_size > 8064:
                    break
                original_sample = all_frames[i:i+window_size]
                subject_trials.append(np.array(original_sample, dtype=np.float32))
                subject_labels.append(int(classes[trial]))
                
                # Augment the sample
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

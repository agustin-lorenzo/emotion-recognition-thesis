import fcwt
import numpy as np
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

# helper method for adding Gaussian noise
def add_gaussian_noise(signal, snr_db=5):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, math.sqrt(noise_power), signal.shape)
    return signal + noise

all_samples = []
all_labels = []

# loop for 32 subjects
for subject in range(32):
    file_path = f"data_preprocessed_python/s{subject + 1:02}.dat"
    x = pickle.load(open(file_path, 'rb'), encoding='latin1')
    data = x['data']
    labels = x['labels']
    relevant_channels = data[:, :32, :]
    relevant_labels = labels[:, :2]
    classes = []

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

    # CWT parameters
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
            total_cwt[start:end, :] = np.abs(current_cwt) # ** 2 # square coefficients to get energy
            
        # average cwt values down by a factor of 4
        total_cwt = total_cwt.reshape(32 * fn, NUM_FRAMES, 4).mean(axis=2)
        
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
        for i in range(0, NUM_FRAMES, SEG_LENGTH):
            if i + SEG_LENGTH > NUM_FRAMES:
                break
            segment = all_frames[i:i+SEG_LENGTH]
            clip = segment[::STRIDE]              
            subject_trials.append(np.array(clip, dtype=np.float32))
            subject_labels.append(int(classes[trial]))
            
            # add augmented samples if needed, set rep_factor to 0 otherwise
            for _ in range(REP_FACTOR - 1):
                augmented_sample = []
                for frame in clip:
                    noisy_frame = add_gaussian_noise(frame, SNR)
                    augmented_sample.append(noisy_frame)
                subject_trials.append(np.array(augmented_sample, dtype=np.float32)) # TODO: normalize correctly if using augmentation later
                subject_labels.append(int(classes[trial]))
    
    all_samples.extend(subject_trials)    
    all_labels.extend(subject_labels)
    
    
    print(f"Processed subject {subject+1}: total samples so far = {len(all_samples)}", end='\r')
np.savez("all_deap_cwt_data.npz",
         samples=np.array(all_samples, dtype=np.float32),
         labels=np.array(all_labels, dtype=np.uint8))
print("\nData saved to all_deap_cwt_data.npz")
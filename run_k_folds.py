import os
import math
import random
import h5py
import pandas as pd
import numpy as np
import fcwt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import subprocess
import gc
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# initialize constant variables
SNR = 5                # signal-to-noise ratio
REP_FACTOR = 100       # total samples per original (1 original + REP_FACTOR-1 augmented copies)
WINDOW_SIZE = 64       # 2-second samples at 128 Hz -> 256 frames -> averaged to 64 frames
VAL_PROB = 5/80        # prob that instance in test set goes to val set, 5/80 with 5 folds leaves   
                       # 75% of data for training each fold

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
    sample_list[count] = np.uint8(sample)
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

# alternate helper function for standardizing/normalizing, then scaling to 0-255 for uint8
def scale_to_uint8(all_frames):
    sample = np.array(all_frames)
    min_val = np.min(sample)
    max_val = np.max(sample)

    if max_val - min_val < 1e-8: # Handle constant input (avoid division by zero)
        return np.full_like(sample, 128, dtype=np.uint8)
    # scale to [0, 1] and then scale to [0, 255]
    normalized = (sample - min_val) / (max_val - min_val)
    sample_uint8 = (normalized * 255).astype(np.uint8)
    return sample_uint8

# helper function for getting sample video from 128 channel .csv rows for one trial
def get_sample(trial_rows, augmentation=False, noise_vectors=None):
    total_cwt = np.zeros((128 * fn, 256)) # image containing cwt values for all channels, change width depending on video/picture data
    # iterate through channels to get 2D CWT image
    for channel, (_, row) in enumerate(trial_rows.iterrows()):
        signal_cols = [str(i) for i in range(1, 257)] # change width depending on wideo/piture data
        signal = row[signal_cols].to_numpy()
        if augmentation:
            if noise_vectors is not None:
                signal = signal + noise_vectors[channel]
            else:
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
    # sample = normalize_sample(all_frames)
    sample = scale_to_uint8(all_frames)
    return sample

print("opening dataset...")
df = pd.read_csv('data/preprocessed_picture.csv')
print("dataset opened.")

# create noise vectors for each participant 
# ensures same noise is applied to each participant's samples similarly, reduces number of times noise is calculated
precomputed_noise = {}
participants = df['par_id'].unique()
for participant in participants:
    participant_data = df[df['par_id'] == participant]
    # get a representative trial by taking a random Neutral stimulus from the participant
    neutral_trials = participant_data[participant_data['Stim_cat'] == 'Neutral']
    rep_stimulus = neutral_trials['Stim_name'].sample(n=1).iloc[0]
    rep_trial = participant_data[participant_data['Stim_name'] == rep_stimulus]
    noise_vectors = {}
    signal_cols = [str(i) for i in range(1, 257)]
    for channel, (_, row) in enumerate(rep_trial.iterrows()):
        signal = row[signal_cols].to_numpy()
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (SNR / 10))
        noise_std = math.sqrt(noise_power)
        noise_vector = np.random.normal(0, noise_std, signal.shape)
        noise_vectors[channel] = noise_vector
    precomputed_noise[participant] = noise_vectors

# get all trials from the original .csv file
trials_list = []
for participant in df['par_id'].unique():
    participant_rows = df[df['par_id'] == participant]
    for stimulus in participant_rows['Stim_name'].unique():
        trial_rows = participant_rows[participant_rows['Stim_name'] == stimulus]
        cls = trial_rows.iloc[0]["class"]
        trials_list.append((trial_rows, int(cls)))

print(f"Total trials extracted: {len(trials_list)}")

# create k folds
labels = [t[1] for t in trials_list]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
    fold += 1
    # create val set indicies
    train_idx_final = []
    val_idx = []
    for i in train_idx:
        if random.random() < VAL_PROB:
            val_idx.append(i)
        else:
            train_idx_final.append(i)
    train_idx = train_idx_final
    
    print(f"\nprocessing fold {fold}...")
    # get class weights from training data to avoid leakage from test data
    train_labels = [trials_list[i][1] for i in train_idx]
    classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=train_labels)
    weights_file = os.path.abspath(f"data/current_class_weights.npy") # save to "current" file so that training script always reads at same location
    np.save(weights_file, class_weights)
    weights_file = os.path.abspath(f"data/class_weights_fold{fold}.npy") # save fold's weights for record keeping
    np.save(weights_file, class_weights)
    print(f"saved class weights for fold {fold} to {weights_file}")
    
    # creating h5 files to avoid keeping all data in memory at once
    train_file_path = os.path.abspath("data/private_train_data.h5")
    val_file_path = os.path.abspath("data/private_val_data.h5")
    test_file_path = os.path.abspath("data/private_test_data.h5")
    train_file = h5py.File(train_file_path, "w")
    val_file = h5py.File(val_file_path, "w")
    test_file = h5py.File(test_file_path, "w")
    
    train_samples = train_file.create_dataset(
        "samples", shape=(0, WINDOW_SIZE, 128, 128),
        maxshape=(None, WINDOW_SIZE, 128, 128), dtype=np.uint8, chunks=True
    )
    train_labels = train_file.create_dataset(
        "labels", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=True
    )
    val_samples = val_file.create_dataset(
        "samples", shape=(0, WINDOW_SIZE, 128, 128),
        maxshape=(None, WINDOW_SIZE, 128, 128), dtype=np.uint8, chunks=True
    )
    val_labels = val_file.create_dataset(
        "labels", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=True
    )
    test_samples = test_file.create_dataset(
        "samples", shape=(0, WINDOW_SIZE, 128, 128),
        maxshape=(None, WINDOW_SIZE, 128, 128), dtype=np.uint8, chunks=True
    )
    test_labels = test_file.create_dataset(
        "labels", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=True
    )
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    # get cwt data from training split and add augmented samples
    for idx in train_idx:
        trial_rows, cls = trials_list[idx]
        sample = get_sample(trial_rows, augmentation=False)
        add_to_h5(train_samples, train_labels, train_count, sample, cls)
        train_count += 1
        # get augmented samples based on precomputed noise vectors for each participant
        participant = trial_rows.iloc[0]['par_id']
        noise_vectors = precomputed_noise.get(participant, None)
        print(f"\tTrain samples so far: {train_count}, Test samples so far: {test_count}", end='\r') 
        for _ in range(REP_FACTOR - 1):
            augmented_sample = get_sample(trial_rows, augmentation=True, noise_vectors=noise_vectors)
            add_to_h5(train_samples, train_labels, train_count, augmented_sample, cls)
            train_count += 1
            print(f"\tTrain samples so far: {train_count}, Val samples so far: {val_count}, Test samples so far: {test_count}", end='\r')
    
    print("\ntraining samples augmented and saved.")
    # get cwt data from val split with no augmentation
    for idx in val_idx:
        trial_rows, cls = trials_list[idx]
        sample = get_sample(trial_rows, augmentation=False)
        add_to_h5(val_samples, val_labels, val_count, sample, cls)
        val_count += 1
        print(f"\tTrain samples so far: {train_count}, Val samples so far: {val_count}, Test samples so far: {test_count}", end='\r')
    
    # get cwt data from test split with no augmentation
    for idx in test_idx:
        trial_rows, cls = trials_list[idx]
        sample = get_sample(trial_rows, augmentation=False)
        add_to_h5(test_samples, test_labels, test_count, sample, cls)
        test_count += 1
        print(f"\tTrain samples so far: {train_count}, Val samples so far: {val_count}, Test samples so far: {test_count}", end='\r')
    
    train_file.close()
    test_file.close()
    
    print(f"Fold {fold} done: {train_count} training samples, {test_count} testing samples.")
    # run training script with current fold's datasets
    print(f"Starting training for fold {fold}...")
    subprocess.run(["python", "-u", "pretrain_and_test.py"])
    
    # delete current fold's datasets
    os.remove(train_file_path)
    os.remove(test_file_path)
    gc.collect()

print("K-fold preprocessing and training complete.")

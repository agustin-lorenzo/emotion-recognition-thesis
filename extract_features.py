import pandas as pd
import numpy as np
import fcwt
import gc

# initialize constant variables
# paramaters for calculating cwt, not to be changed
fs = 128 
f0 = 4 # lowest frequency
f1 = 45 # highest frequency
fn = 128 # number of frequencies, match channel number for square frame

print("opening dataset...")
df = pd.read_csv('data/preprocessed_video.csv')
print("dataset opened.")

trials = [] # list of cwt videos for all trials
labels = [] # classes corresponding to all trials

# iterate through participants
for participant in df['par_id'].unique():
    participant_rows = df[df['par_id'] == participant]
    
    print("\ncurrent participant: ", participant)
    # iterate through trials
    for stimulus in participant_rows['stimulus'].unique():
        trial_rows = participant_rows[participant_rows['stimulus'] == stimulus]
        cls = trial_rows.iloc[0]["class"]
        
        total_cwt = np.zeros((128 * fn, 1280)) # image containing cwt values for all channels
        
        # iterate through channels
        for channel, (_, row) in enumerate(trial_rows.iterrows()):
            signal_cols = [str(i) for i in range(1, 1281)]
            signal = row[signal_cols].to_numpy()
            _, current_cwt = fcwt.cwt(signal, fs, f0, f1, fn)
            start = channel * fn
            end = (channel + 1) * fn
            total_cwt[start:end, :] = abs(current_cwt)
        
        # convert 2D time sample x channel-frequency format (x, y) to 3D channel x frequency x time format (x, y, z)
        # each 'frame' in all_frames is a 2D image showing each channel's CWT value for that time sample
        all_frames = []
        for sample in range(1280):
            frame = np.zeros((128, 128))
            for channel in range(128):
                start = channel * fn
                end = (channel + 1) * fn
                frame[:, channel] = total_cwt[start:end, sample]
            
            norm_frame = (frame - frame.min()) / (frame.max() - frame.min()) # normalize frame
            scaled_frame = (norm_frame * 255).astype(np.uint8) # scale to 255 for video
            all_frames.append(scaled_frame) # append 1-channel frame, no rgb
            
        # split trial into 6-second samples as suggested in Arjun et al. 2021
        window_size = 5 * 128
        for i in range(0, 1280, window_size):
            if i + window_size > 1280:
                break
            clip = all_frames[i:i+window_size]
            trials.append(clip) # append on this level for clipping
            labels.append(cls)
        
        print("\tCurrent Number of Trials:", len(trials), end='\r')
        
        # trials.append(all_frames) # append on this level for no clipping
        # labels.append(cls)
        
print("\nsaving features...")
trials_array = np.stack(trials)  # shape: (num_trials, 128, 128, 1280/2)
del trials
gc.collect()

labels_array = np.array(labels)
del labels
gc.collect()

print("\tTrials shape:", trials_array.shape)  # should be (num_trials, 128, 128, 1280/2)
print("\tTrials dtype:", trials_array.dtype)

print("\n\tLabels shape:", labels_array.shape)  # should be (num_trials,)
print("\tLabels dtype:", labels_array.dtype)
np.savez_compressed('data/extracted_features_compressed.npz', trials=trials_array, labels=labels_array)
print("done.")
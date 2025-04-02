import h5py
import numpy as np

# Define the batch size for processing (adjust based on your available memory)
batch_size = 100
file_path = "deap_cwt_data.h5"

with h5py.File(file_path, "r+") as hf:
    samples_dset = hf["samples"]
    n_samples = samples_dset.shape[0]
    
    # First pass: Compute global mean using the training data stored in the file
    total_sum = 0.0
    total_count = 0
    for i in range(0, n_samples, batch_size):
        batch = samples_dset[i:i+batch_size]
        total_sum += batch.sum()
        total_count += batch.size
    global_mean = total_sum / total_count
    
    # Second pass: Compute global standard deviation
    total_sq_diff = 0.0
    for i in range(0, n_samples, batch_size):
        batch = samples_dset[i:i+batch_size]
        total_sq_diff += ((batch - global_mean) ** 2).sum()
    global_variance = total_sq_diff / total_count
    global_std = np.sqrt(global_variance)
    
    print("Global Mean:", global_mean)
    print("Global Std:", global_std)
    
    # Third pass: Standardize the data in-place batch by batch
    for i in range(0, n_samples, batch_size):
        batch = samples_dset[i:i+batch_size]
        standardized_batch = (batch - global_mean) / global_std
        samples_dset[i:i+batch_size] = standardized_batch
        print(f"Standardized batch {i} to {i + batch_size}")

print("In-place standardization complete.")

import h5py
import numpy as np

batch_size = 100  # Adjust based on available memory
train_file_path = "data/deap_train_data.h5"
test_file_path = "data/deap_test_data.h5"

# --- Compute Training Statistics ---
with h5py.File(train_file_path, "r+") as hf_train:
    samples_dset = hf_train["samples"]
    n_samples = samples_dset.shape[0]
    
    total_sum = 0.0
    total_count = 0
    for i in range(0, n_samples, batch_size):
        batch = samples_dset[i:i+batch_size]
        total_sum += batch.sum()
        total_count += batch.size
    train_mean = total_sum / total_count

    total_sq_diff = 0.0
    for i in range(0, n_samples, batch_size):
        batch = samples_dset[i:i+batch_size]
        total_sq_diff += ((batch - train_mean) ** 2).sum()
    train_std = np.sqrt(total_sq_diff / total_count)
    
    print("Training data stats: mean =", train_mean, "std =", train_std)
    
    # Standardize training data in place
    for i in range(0, n_samples, batch_size):
        batch = samples_dset[i:i+batch_size]
        standardized_batch = (batch - train_mean) / train_std
        samples_dset[i:i+batch_size] = standardized_batch
        print(f"Standardized training batch {i} to {i + batch_size}")

# --- Standardize Test Data Using Training Statistics ---
with h5py.File(test_file_path, "r+") as hf_test:
    samples_dset = hf_test["samples"]
    n_samples = samples_dset.shape[0]
    for i in range(0, n_samples, batch_size):
        batch = samples_dset[i:i+batch_size]
        standardized_batch = (batch - train_mean) / train_std
        samples_dset[i:i+batch_size] = standardized_batch
        print(f"Standardized test batch {i} to {i + batch_size}")

print("In-place standardization complete for both training and testing datasets.")

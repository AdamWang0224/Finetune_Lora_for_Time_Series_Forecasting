import h5py
import numpy as np

"""
This script reads the original Lotka-Volterra dataset, splits it into training and test sets,
and saves the results into two separate HDF5 files.

The dataset contains time series of predator and prey populations, shaped (1000, 100, 2),
where:
- 1000: number of independent systems,
- 100: time points per system,
- 2: prey and predator population values.
"""

# Load the original data from the HDF5 file
with h5py.File("./data/lotka_volterra_data.h5", "r") as f:
    trajectories = f["trajectories"][:]  # Shape: (1000, 100, 2)
    time_points = f["time"][:]           # Shape: (100,), shared across all systems

# Method 1: Split the data sequentially (first 900 systems for training, last 100 for testing)
# train_trajectories = trajectories[:900]
# test_trajectories = trajectories[900:]

# Alternative method: Random split to ensure evaluation data is completely disjoint from training data
# Uncomment the following lines to use random splitting instead
np.random.seed(42)  # Set random seed for reproducibility
indices = np.arange(trajectories.shape[0])
np.random.shuffle(indices)
split_index = int(0.9 * len(indices))
train_indices = indices[:split_index]
test_indices = indices[split_index:]
train_trajectories = trajectories[train_indices]
test_trajectories = trajectories[test_indices]

# Save the training data to a new HDF5 file
with h5py.File("./data/lotka_volterra_train.h5", "w") as f_train:
    f_train.create_dataset("trajectories", data=train_trajectories)
    f_train.create_dataset("time", data=time_points)

# Save the test data to a new HDF5 file
with h5py.File("./data/lotka_volterra_test.h5", "w") as f_test:
    f_test.create_dataset("trajectories", data=test_trajectories)
    f_test.create_dataset("time", data=time_points)

print("Data successfully split and saved to 'lotka_volterra_train.h5' (training) and 'lotka_volterra_test.h5' (testing).")

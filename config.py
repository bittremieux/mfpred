random_seed = 42

# Input.
filename_train = "../data/MassSpecGym_train.mgf"
filename_val = "../data/MassSpecGym_val.mgf"
out_dir = "../data/out/"

# Spectrum preprocessing.
min_mz = 50
remove_precursor_tol_mass = 0.02
remove_precursor_tol_mode = "ppm"
min_intensity = 0.01
max_num_peaks = 150
scaling = "root"

# Supported atoms.
vocab = "CHNOPS"

# Model hyperparameters.
d_model = 256
nhead = 8
dim_feedforward = 2048
n_layers = 5
dropout = 0.1
max_atom_cardinality = 100
tau = 1.0

# Training settings.
lr = 1e-3
batch_size = 1024
n_epochs = 100
patience = 10

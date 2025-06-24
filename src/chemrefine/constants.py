# constants.py

# Conversion constants
HARTREE_TO_KCAL_MOL = 2625.49964 / 4.184
R_KCAL_MOL_K = 8.314462618e-3 / 4.184  # kcal/(mol*K)

# Default configuration
DEFAULT_TEMPERATURE = 298.15  # K
DEFAULT_CORES = 32
DEFAULT_MAX_CORES = 32
DEFAULT_ENERGY_WINDOW = 0.5  # Hartree
DEFAULT_BOLTZMANN_PERCENTAGE = 99  # %

# Job handling
JOB_CHECK_INTERVAL = 5  # seconds
CSV_PRECISION = 8  # for float rounding in CSV

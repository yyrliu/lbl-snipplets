# H5 Spectroscopy Analysis Tools

This repository contains tools for analyzing UV-Vis and PL spectroscopy data from H5 files.

## Files Structure

### Core Module
- **`h5_analysis.py`** - Main module containing all core functions for:
  - H5 file discovery and filtering
  - Data reading and preprocessing
  - UV-Vis and PL spectroscopy analysis
  - Plotting and visualization
  - Batch processing capabilities

### Notebooks
- **`h5-plot.ipynb`** - Interactive Jupyter notebook for exploratory analysis
  - Uses functions from `h5_analysis.py`
  - Includes advanced analysis functions
  - Contains examples and visualizations

### Examples
- **`example_usage.py`** - Standalone script demonstrating module usage
  - Shows how to use the module in Python scripts
  - Includes batch processing examples

## Quick Start

### In Python Scripts
```python
from h5_analysis import discover_h5_files, analyze_h5_file

# Discover H5 files
data_dirs = ["path/to/your/CBox/folder"]
h5_files = discover_h5_files(data_dirs)

# Analyze a single file
result = analyze_h5_file(h5_files[0], plot_results=True)
```

### In Jupyter Notebooks
```python
from h5_analysis import *

# All functions are available
files = discover_h5_files(["your/data/path"])
results = analyze_multiple_files(files, max_files=5)
summary_df = create_summary_dataframe(results)
```

## Key Functions

### File Discovery
- `discover_h5_files(data_dirs)` - Find and filter H5 files
- `filter_h5_files(h5_files)` - Filter to keep highest run numbers

### Data Processing
- `read_h5_file(file_path)` - Read H5 file and extract data
- `process_uv_vis_data(wl_data, wls_range=None)` - Process UV-Vis data
- `process_pl_data(pl_data, wls_range=None)` - Process PL data

### Analysis
- `analyze_h5_file(file_path, ...)` - Complete analysis of single file
- `analyze_multiple_files(file_list, ...)` - Batch analysis
- `create_summary_dataframe(results)` - Create summary tables

### Visualization
- `plot_uv_vis_spectra(uv_vis_data, sample_id)` - Plot UV-Vis spectra
- `plot_pl_spectra(pl_data, sample_id)` - Plot PL spectra
- `plot_sample_photo(photo, sample_id)` - Plot sample photos

### Utility Functions
- `wavelength_to_energy(wavelength_nm)` - Convert wavelength to energy
- `apply_jacobian(signal, wavelength_nm)` - Apply Jacobian transformation
- `find_nearest(array, value)` - Find nearest array value

## Data Structure

The module expects H5 files with the following structure:
```
measurement/spec_run/
├── photo                    # Sample photos
├── wl_dark_spectrum        # UV-Vis dark spectrum
├── wl_spectra              # UV-Vis sample spectra
├── wl_ref_spectrum         # UV-Vis reference spectrum
├── wl_wls                  # UV-Vis wavelength axis
├── pl_dark_spectrum        # PL dark spectrum
├── pl_spectra              # PL sample spectra
├── pl_wls                  # PL wavelength axis
└── ...
```

## Configuration

Default data directories can be set in `h5_analysis.py`:
```python
DEFAULT_DATA_DIRS = [
    r"g:\My Drive\LPS\20250709_S_MeOMBAI_prestudy_2\CBox"
]
```

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- h5py
- scipy
- pathlib
- re

## Features

- ✅ Automatic file discovery with run number filtering
- ✅ UV-Vis transmission and absorption analysis
- ✅ Multi-position PL spectroscopy processing
- ✅ Energy scale conversion with Jacobian transformation
- ✅ Batch processing capabilities
- ✅ Comprehensive visualization
- ✅ Error handling and data validation
- ✅ Customizable wavelength ranges
- ✅ Summary statistics and export

## Usage Examples

See `example_usage.py` for a complete example, or run the Jupyter notebook `h5-plot.ipynb` for interactive analysis.

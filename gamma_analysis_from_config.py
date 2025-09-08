import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import AutoMinorLocator
from plot_helper import (
    get_linestyle_factory, 
    get_color_factory, 
    filter_columns, 
    fit_peak_fwhm,
    get_label_from_mapping
)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_gamma_files(data_dirs, pattern):
    """Find gamma scan files based on pattern"""
    files = []
    for d in data_dirs:
        files.extend(Path(d).glob(pattern))
    return files


def load_gamma_data(gamma_files, label_mapping):
    """Load and process gamma scan data with reindexing like regular XRD"""
    dfs = []
    labels = []
    
    # Group files by gamma angle
    gamma_groups = {}
    for gamma_file in gamma_files:
        # Extract gamma angle from filename
        stem = gamma_file.stem
        gamma_match = re.search(r'_gamma_(\d+)', stem)
        if gamma_match:
            gamma_angle = gamma_match.group(1)
        else:
            # For testing purposes, if no gamma angle found, use "12" as default
            gamma_angle = "12"
            print(f"Note: No gamma angle found in {stem}, using default gamma_12")
            
        if gamma_angle not in gamma_groups:
            gamma_groups[gamma_angle] = []
        gamma_groups[gamma_angle].append(gamma_file)
    
    # Process each gamma angle group
    for gamma_angle, files in gamma_groups.items():
        # Sort files by name and reindex as gamma_{angle}_1, gamma_{angle}_2, ...
        files_sorted = sorted(files, key=lambda f: f.name)
        
        for idx, gamma_file in enumerate(files_sorted, 1):
            reindexed_name = f"gamma_{gamma_angle}_{idx}.xy"
            
            # Use the existing label mapping with the gamma prefix
            label = get_label_from_mapping(f"gamma_{gamma_angle}_{idx}", label_mapping)
            
            print(f"{gamma_file.name} -> {reindexed_name} | label: {label}")
            
            # Read gamma data (no wavelength conversion needed)
            # For testing with regular XRD files, use 2theta as gamma
            try:
                df = pd.read_csv(gamma_file, sep='\t', comment='#', names=['gamma', 'intensity'])
            except Exception:
                # Fallback for different file formats
                df = pd.read_csv(gamma_file, sep='\t', header=None, names=['gamma', 'intensity'])
                
            df = df.set_index('gamma')
            
            # Filter reasonable gamma range (adjust for testing with 2theta data)
            df = df[(df.index > -150) & (df.index < 150)] if 'gamma' in str(gamma_file) else df[(df.index > 4) & (df.index < 50)]
            
            dfs.append(df['intensity'])
            labels.append(label)
    
    if not dfs:
        print("Warning: No gamma data files found!")
        return pd.DataFrame()
        
    df_all = pd.concat(dfs, axis=1)
    df_all.columns = labels
    return df_all


def perform_peak_fitting(df, peak_center_guess, cols):
    """Perform peak fitting analysis and return results"""
    print("\n=== Peak Fitting Results ===")
    fwhm_results = []
    fit_results_full = {}
    
    for col in cols:
        df_to_plot = df[[col]].dropna()
        x_data = df_to_plot.index.values
        y_data = df_to_plot[col].values
        
        # Fit peak and get FWHM
        result = fit_peak_fwhm(x_data, y_data, peak_center_guess)
        
        if result['success']:
            fwhm_results.append({
                'sample': col,
                'fwhm': result['fwhm'],
                'center': result['center'],
                'r_squared': result['r_squared']
            })
            fit_results_full[col] = result
            print(f"{col:30s} | FWHM: {result['fwhm']:.3f}° | Center: {result['center']:.2f}° | R²: {result['r_squared']:.4f}")
        else:
            print(f"{col:30s} | Fit failed: {result['error']}")
    
    return fwhm_results, fit_results_full


def plot_gamma_analysis(df, plot_cfg, data_dir=None):
    """Create gamma analysis plots with FWHM fitting"""
    cols = filter_columns(df, plot_cfg.get('filters', []))
    
    # Get options
    options = plot_cfg.get('options', {})
    xlim = options.get('xlim', [-150, -30])
    xlim_zoom = options.get('xlim_zoom', [-115, -65])
    ylim = options.get('ylim', [10, None])
    gaussian_sigma = options.get('gaussian_filter_sigma', 2)
    show_fwhm = options.get('show_fwhm_in_legend', True)
    peak_pos = plot_cfg.get('peak_pos', -90)
    
    # Perform peak fitting
    fwhm_results, fit_results_full = perform_peak_fitting(df, peak_pos, cols)
    fwhm_lookup = {r['sample']: r['fwhm'] for r in fwhm_results}
    
    # Create the plot
    fig, (ax, ax_zoom) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), dpi=300, layout="tight")
    
    # Create styling functions from config
    style_config = plot_cfg.get("style", {})
    get_color = get_color_factory(**style_config.get("linecolor", {}))
    get_linestyle = get_linestyle_factory(**style_config.get("linestyle", {}))
    
    # Plot data
    for i, col in enumerate(sorted(cols)):
        df_to_plot = df[[col]].dropna()
        color = get_color(col)
        linestyle = get_linestyle(col)
        
        # Create label with FWHM if available and requested
        if show_fwhm and col in fwhm_lookup:
            label = f"{col} (FWHM: {fwhm_lookup[col]:.3f}°)"
        else:
            label = col
        
        # Plot smoothed data on main plot
        ax.plot(
            df_to_plot.index,
            gaussian_filter1d(df_to_plot[col], gaussian_sigma),
            label=label,
            linestyle=linestyle,
            color=color,
            linewidth=1.5,
            alpha=0.8
        )
        
        # Plot data on zoom plot with offset for visibility
        ax_zoom.plot(
            df_to_plot.index,
            df_to_plot[col] * (2 ** i),
            label=col,
            linestyle=linestyle,
            color=color,
            linewidth=1.5,
            alpha=0.8
        )
        
        # Plot fit on zoom plot if available
        if col in fit_results_full:
            fit_result = fit_results_full[col]
            ax_zoom.plot(
                fit_result['x_fit'],
                fit_result['y_pred'] * (2 ** i),
                '--',
                color=color,
                alpha=0.9,
                linewidth=2.5
            )
    
    # Style both axes
    for axis in [ax, ax_zoom]:
        axis.set_yscale("log")
        axis.set_xlabel("Gamma angle (degrees, Co K$\\alpha$)", fontsize=12)
        axis.set_ylabel("Intensity (a.u.)", fontsize=12)
        axis.set_ylim(*ylim)
        axis.minorticks_on()
        minor_locator = AutoMinorLocator(10)
        axis.xaxis.set_minor_locator(minor_locator)
        axis.grid(True, which="major", alpha=0.3, linewidth=0.8)
        axis.grid(True, which="minor", alpha=0.15, linewidth=0.4)
        axis.tick_params(labelsize=10)
    
    # Set specific limits
    ax.set_xlim(*xlim)
    ax_zoom.set_xlim(*xlim_zoom)
    ax_zoom.set_title("Peak Fitting Region (-110° to -70°)", fontsize=12, pad=10)
    ax.set_title("Gamma Analysis", fontsize=12, pad=10)
    
    # Add legends
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=1)
    ax_zoom.legend(loc='upper right', fontsize=8, framealpha=0.95)
    
    # Print summary statistics
    if fwhm_results:
        fwhm_values = [r['fwhm'] for r in fwhm_results]
        center_values = [r['center'] for r in fwhm_results]
        print(f"\n=== Summary (n={len(fwhm_results)} successful fits) ===")
        print(f"FWHM range: {min(fwhm_values):.3f}° to {max(fwhm_values):.3f}°")
        print(f"Mean FWHM: {np.mean(fwhm_values):.3f}° ± {np.std(fwhm_values):.3f}°")
        print(f"Peak center range: {min(center_values):.2f}° to {max(center_values):.2f}°")
        print(f"Mean peak center: {np.mean(center_values):.2f}° ± {np.std(center_values):.2f}°")
    
    # Save the plot
    if data_dir:
        output_path = Path(data_dir) / plot_cfg['output_file']
    else:
        output_path = Path(plot_cfg['output_file'])
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Check if gamma_analysis section exists
    if 'gamma_analysis' not in cfg:
        print("No gamma_analysis section found in config")
        return
    
    gamma_cfg = cfg['gamma_analysis']
    
    # Find gamma files
    gamma_files = find_gamma_files(
        gamma_cfg['data_dirs'], 
        gamma_cfg['file_pattern']
    )
    
    if not gamma_files:
        print(f"No gamma files found with pattern: {gamma_cfg['file_pattern']}")
        return
    
    print(f"Found {len(gamma_files)} gamma files")
    
    # Load data
    df = load_gamma_data(gamma_files, gamma_cfg['label_mapping'])
    
    # Use the first data directory for saving plots
    data_dir = gamma_cfg['data_dirs'][0] if gamma_cfg['data_dirs'] else None
    
    # Process each plot configuration
    for plot_cfg in gamma_cfg['plots']:
        plot_gamma_analysis(df, plot_cfg, data_dir)


if __name__ == '__main__':
    main()

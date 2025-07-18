"""
Example usage of the h5_analysis module for spectroscopy data analysis.

This script demonstrates how to use the h5_analysis module to:
1. Discover H5 files in directories
2. Analyze individual files
3. Perform batch analysis
4. Create summary tables
"""

from h5_analysis import (
    discover_h5_files, analyze_h5_file, analyze_multiple_files, 
    create_summary_dataframe
)

def main():
    """Main example function."""
    print("H5 Analysis Example")
    print("==================")
    
    # 1. Discover H5 files
    data_dirs = [
        r"g:\My Drive\LPS\20250709_S_MeOMBAI_prestudy_2\CBox"
        # Add more directories as needed
    ]
    
    h5_files = discover_h5_files(data_dirs)
    
    if len(h5_files) == 0:
        print("No H5 files found. Please check your data directories.")
        return
    
    print(f"\nFound {len(h5_files)} H5 files ready for analysis.")
    
    # 2. Analyze a single file
    print("\n" + "="*50)
    print("SINGLE FILE ANALYSIS")
    print("="*50)
    
    # Analyze first file with plots
    result = analyze_h5_file(h5_files[0], plot_results=True)
    
    # 3. Batch analysis (without plots)
    print("\n" + "="*50)
    print("BATCH ANALYSIS")
    print("="*50)
    
    # Analyze multiple files
    batch_results = analyze_multiple_files(h5_files, max_files=5)
    
    # Create summary table
    summary_df = create_summary_dataframe(batch_results)
    print("\nSUMMARY TABLE:")
    print(summary_df.to_string(index=False))
    
    # 4. Save summary to CSV
    summary_df.to_csv('h5_analysis_summary.csv', index=False)
    print("\nSummary saved to 'h5_analysis_summary.csv'")
    
    # 5. Example of custom wavelength ranges
    print("\n" + "="*50)
    print("CUSTOM WAVELENGTH RANGES")
    print("="*50)
    
    # Analyze with custom UV-Vis and PL ranges
    custom_result = analyze_h5_file(
        h5_files[0], 
        plot_results=False,
        uv_vis_range=(400, 700),  # UV-Vis range: 400-700 nm
        pl_range=(650, 950)       # PL range: 650-950 nm
    )
    
    print("Analysis complete with custom wavelength ranges!")

if __name__ == "__main__":
    main()

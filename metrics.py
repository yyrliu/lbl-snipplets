"""
Metrics Pipeline

This module calculates heterogeneity scores for H5 files from CBox measurements
and writes them to the metrics/heterogeneity dataset in the HDF5 structure.

The heterogeneity score is calculated from the image data at /results/cbox/photo/image
and combines three metrics:
- Entropy (10% weight) - Measures randomness/disorder in pixel intensity
- Standard Deviation (20% weight) - Measures overall intensity dispersion
- Radial Extrema Variance (70% weight) - Measures intensity variations in concentric rings

The calculated score is written to metrics/heterogeneity in each H5 file.
"""

import h5py
import argparse
import sys
from pathlib import Path
from heterogeneity import optimize_PL_image


def add_heterogeneity_metrics(h5_file_path, verbose=True):
    """
    Calculate and add heterogeneity score to h5 file.
    Reads image from /results/cbox/photo/image and writes score to metrics/heterogeneity

    Args:
        h5_file_path (str): Path to the h5 file
        verbose (bool): Enable verbose output

    Returns:
        float or None: Heterogeneity score if successful, None if failed
    """
    try:
        with h5py.File(h5_file_path, "r+") as hf:
            # Read the image data from results/cbox/photo/image
            if "results/cbox/photo/image" not in hf:
                if verbose:
                    print(
                        f"Warning: 'results/cbox/photo/image' not found in {h5_file_path}"
                    )
                return None

            image_data = hf["results/cbox/photo/image"][:]
            if verbose:
                print(f"Found image data with shape: {image_data.shape}")

            # Calculate heterogeneity score
            heterogeneity_score = optimize_PL_image(image_data)
            if verbose:
                print(f"Calculated heterogeneity score: {heterogeneity_score:.6f}")

            # Create metrics group if it doesn't exist
            if "metrics" not in hf:
                metrics_group = hf.create_group("metrics")
                if verbose:
                    print("Created 'metrics' group")
            else:
                metrics_group = hf["metrics"]
                if verbose:
                    print("Using existing 'metrics' group")

            # Write heterogeneity score to metrics/heterogeneity
            if "heterogeneity" in metrics_group:
                # Update existing dataset
                del metrics_group["heterogeneity"]
                if verbose:
                    print("Updated existing heterogeneity dataset")

            metrics_group.create_dataset("heterogeneity", data=heterogeneity_score)
            if verbose:
                print(f"Added heterogeneity score: {heterogeneity_score:.6f}")

            return heterogeneity_score

    except Exception as e:
        if verbose:
            print(f"Error processing {h5_file_path}: {str(e)}")
        return None


def batch_add_heterogeneity_metrics(h5_file_paths, verbose=True):
    """
    Process multiple h5 files to add heterogeneity metrics

    Args:
        h5_file_paths (list): List of h5 file paths
        verbose (bool): Enable verbose output

    Returns:
        dict: Dictionary mapping file paths to heterogeneity scores
    """
    results = {}
    successful = 0
    failed = 0

    for i, h5_file_path in enumerate(h5_file_paths, 1):
        if verbose:
            print(f"\n[{i}/{len(h5_file_paths)}] Processing: {Path(h5_file_path).name}")

        score = add_heterogeneity_metrics(h5_file_path, verbose=verbose)
        results[h5_file_path] = score

        if score is not None:
            successful += 1
        else:
            failed += 1

    if verbose:
        print("\n=== BATCH PROCESSING SUMMARY ===")
        print(f"Total files processed: {len(h5_file_paths)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

    return results


def verify_heterogeneity_metrics(h5_file_path, verbose=True):
    """
    Verify that heterogeneity metrics were written correctly

    Args:
        h5_file_path (str): Path to the h5 file
        verbose (bool): Enable verbose output

    Returns:
        float or None: Heterogeneity score if found, None otherwise
    """
    try:
        with h5py.File(h5_file_path, "r") as hf:
            if "metrics/heterogeneity" in hf:
                score = hf["metrics/heterogeneity"][()]
                if verbose:
                    print(
                        f"Heterogeneity score in {Path(h5_file_path).name}: {score:.6f}"
                    )
                return score
            else:
                if verbose:
                    print(
                        f"No heterogeneity metrics found in {Path(h5_file_path).name}"
                    )
                return None

    except Exception as e:
        if verbose:
            print(f"Error reading {h5_file_path}: {str(e)}")
        return None


def discover_h5_files(directories, verbose=True):
    """
    Discover all H5 files in the given directories

    Args:
        directories (list): List of directory paths
        verbose (bool): Enable verbose output

    Returns:
        list: List of H5 file paths
    """
    h5_files = []

    for directory in directories:
        dir_path = Path(directory)

        # Look for H5 files in the specified directory
        h5_files_found = list(dir_path.glob("*.h5"))

        # Also look recursively for H5 files
        h5_files_recursive = list(dir_path.glob("**/*.h5"))

        # Combine and deduplicate
        directory_files = list(set(h5_files_found + h5_files_recursive))

        if verbose and directory_files:
            print(f"Found {len(directory_files)} H5 files in {directory}")

        h5_files.extend([str(f) for f in directory_files])

    return h5_files


def check_metrics_status(directories, verbose=True):
    """
    Check which H5 files already have heterogeneity metrics

    Args:
        directories (list): List of directory paths
        verbose (bool): Enable verbose output

    Returns:
        dict: Status summary
    """
    h5_files = discover_h5_files(directories, verbose=False)

    with_metrics = []
    without_metrics = []
    invalid_files = []

    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, "r") as hf:
                if "metrics/heterogeneity" in hf:
                    with_metrics.append(h5_file)
                else:
                    without_metrics.append(h5_file)
        except Exception:
            invalid_files.append(h5_file)

    if verbose:
        print("=== METRICS STATUS CHECK ===")
        print(f"Total H5 files: {len(h5_files)}")
        print(f"With heterogeneity metrics: {len(with_metrics)}")
        print(f"Without heterogeneity metrics: {len(without_metrics)}")
        print(f"Invalid/inaccessible files: {len(invalid_files)}")

        if without_metrics:
            print("\nFiles without metrics:")
            for f in without_metrics[:5]:  # Show first 5
                print(f"  {Path(f).name}")
            if len(without_metrics) > 5:
                print(f"  ... and {len(without_metrics) - 5} more")

    return {
        "total_files": len(h5_files),
        "with_metrics": len(with_metrics),
        "without_metrics": len(without_metrics),
        "invalid_files": len(invalid_files),
        "files_with_metrics": with_metrics,
        "files_without_metrics": without_metrics,
        "invalid_file_list": invalid_files,
    }


def main():
    """
    Main function to run the metrics calculation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Metrics Pipeline - Calculate heterogeneity scores for H5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Structure Expected:
  {dir_path}/
  └── *.h5                  # H5 files with results/cbox/photo/image structure
  
Required H5 Structure:
  Each H5 file must contain:
  └── results/cbox/photo/image  # Image data for heterogeneity calculation
  
Output:
  Writes heterogeneity score to:
  └── metrics/heterogeneity     # Scalar heterogeneity score

Example usage:
  # Check metrics status only (see which files have/don't have metrics)
  uv run metrics.py "G:\\My Drive\\LPS\\LPS-1" --check-only
  
  # Calculate metrics for all files  
  uv run metrics.py "G:\\My Drive\\LPS\\LPS-1"
  
  # Process only first 3 files (useful for testing)
  uv run metrics.py "G:\\My Drive\\LPS\\LPS-1" --max-files 3
  
  # Process multiple directories
  uv run metrics.py "G:\\My Drive\\LPS\\20250709_S_MeOMBAI_prestudy_1" "G:\\My Drive\\LPS\\20250709_S_MeOMBAI_prestudy_2"
  
  # Verify existing metrics
  uv run metrics.py "G:\\My Drive\\LPS\\LPS-1" --verify-only
        """,
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more base directories containing H5 files",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Disable verbose output"
    )

    parser.add_argument(
        "--check-only",
        "-c",
        action="store_true",
        help="Only check which files have/don't have heterogeneity metrics (no processing)",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing heterogeneity metrics (no processing)",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit processing to first N files (useful for testing). Only applies when processing files",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing heterogeneity metrics (default: skip files with existing metrics)",
    )

    args = parser.parse_args()

    # Handle verbose flag
    verbose = args.verbose and not args.quiet

    # Validate directories
    valid_dirs = []
    for dir_path in args.directories:
        path = Path(dir_path)
        if not path.exists():
            print(f"Error: Directory does not exist: {dir_path}")
            sys.exit(1)
        if not path.is_dir():
            print(f"Error: Path is not a directory: {dir_path}")
            sys.exit(1)
        valid_dirs.append(str(path.resolve()))

    if verbose:
        print(f"Processing directories: {valid_dirs}")
        print("Looking for H5 files in:")
        print("  **/*.h5               # H5 files with proper results structure")
        print()

    try:
        if args.check_only:
            print("=== METRICS STATUS CHECK MODE ===")
            print("Checking which files have heterogeneity metrics...")
            print()

            status = check_metrics_status(valid_dirs, verbose=verbose)

            if status["total_files"] > 0:
                print("\nStatus check completed successfully!")
                if status["without_metrics"] > 0:
                    print(
                        "Run without --check-only to calculate metrics for remaining files."
                    )
            else:
                print("No H5 files found to check.")

        elif args.verify_only:
            print("=== METRICS VERIFICATION MODE ===")
            print("Verifying existing heterogeneity metrics...")
            print()

            h5_files = discover_h5_files(valid_dirs, verbose=verbose)

            if not h5_files:
                print("No H5 files found to verify.")
                return

            verified_count = 0
            for h5_file in h5_files:
                score = verify_heterogeneity_metrics(h5_file, verbose=verbose)
                if score is not None:
                    verified_count += 1

            print(
                f"\nVerification completed: {verified_count}/{len(h5_files)} files have heterogeneity metrics"
            )

        else:
            # Process files
            h5_files = discover_h5_files(valid_dirs, verbose=verbose)

            if not h5_files:
                print("No H5 files found to process.")
                print("Please check:")
                print("1. H5 files exist in CBox or preprocessed subdirectories")
                print("2. Directory structure matches expected layout")
                print("3. Run with --check-only flag to verify metrics status")
                return

            # Filter files if not forcing overwrite
            if not args.force:
                files_to_process = []
                skipped_count = 0

                for h5_file in h5_files:
                    try:
                        with h5py.File(h5_file, "r") as hf:
                            if "metrics/heterogeneity" not in hf:
                                files_to_process.append(h5_file)
                            else:
                                skipped_count += 1
                    except Exception:
                        # If we can't read the file, try to process it anyway
                        files_to_process.append(h5_file)

                if verbose and skipped_count > 0:
                    print(
                        f"Skipping {skipped_count} files that already have heterogeneity metrics"
                    )
                    print("Use --force to overwrite existing metrics")
                    print()

                h5_files = files_to_process

            # Apply max_files limit
            if args.max_files and len(h5_files) > args.max_files:
                if verbose:
                    print(
                        f"Limiting to first {args.max_files} files (out of {len(h5_files)} total)"
                    )
                h5_files = h5_files[: args.max_files]

            if not h5_files:
                print("No files to process (all files already have metrics).")
                print("Use --force to overwrite existing metrics.")
                return

            # Process the files
            results = batch_add_heterogeneity_metrics(h5_files, verbose=verbose)

            # Summary
            successful_files = [f for f, score in results.items() if score is not None]
            failed_files = [f for f, score in results.items() if score is None]

            if verbose:
                print("\n=== FINAL SUMMARY ===")
            print(f"Successfully processed {len(successful_files)} H5 files")

            if failed_files:
                print(f"Failed to process {len(failed_files)} files")
                if verbose:
                    print("Failed files:")
                    for f in failed_files:
                        print(f"  {Path(f).name}")

            if not successful_files and not failed_files:
                print("No files were processed.")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

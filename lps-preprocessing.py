"""
LPS Preprocessing Pipeline

This module processes H5 files from CBox measurements and integrates associated CD and XRD data
into a unified HDF5 structure for comprehensive analysis.

OPTIMIZATION: Photo data (/results/cbox/photo/image) is now softlinked to /measurement/spec_run/adj_photo
instead of being copied, eliminating redundancy and saving storage space.

H5 Data Structure (results group):
=================================

results/                          # New processed results group
├── cbox/                        # CBox spectroscopy results
│   ├── uv_vis/
│   │   ├── wavelength           # 1D array [n_wavelengths]
│   │   ├── energy               # 1D array [n_wavelengths]
│   │   ├── absorption           # 2D array [N, n_wavelengths] (individual spectra)
│   │   ├── absorption_avg       # 1D array [n_wavelengths] (averaged spectrum)
│   │   ├── absorption_jacobian  # 2D array [N, n_wavelengths] (individual with Jacobian)
│   │   ├── absorption_jacobian_avg # 1D array [n_wavelengths] (averaged with Jacobian)
│   │   ├── transmission         # 2D array [N, n_wavelengths] (individual spectra)
│   │   ├── transmission_avg     # 1D array [n_wavelengths] (averaged spectrum)
│   │   └── num_spectra          # Scalar (number of individual spectra, typically 8)
│   │
│   ├── pl/
│   │   ├── wavelength           # 1D array [n_wavelengths]
│   │   ├── energy               # 1D array [n_wavelengths]
│   │   ├── intensity            # 2D array [N, n_wavelengths] (individual spectra)
│   │   ├── intensity_avg        # 1D array [n_wavelengths] (averaged spectrum)
│   │   ├── intensity_jacobian   # 2D array [N, n_wavelengths] (individual with Jacobian)
│   │   ├── intensity_jacobian_avg # 1D array [n_wavelengths] (averaged with Jacobian)
│   │   └── num_spectra          # Scalar (number of individual spectra)
│   │
│   └── photo/
│       ├── image                # Softlink to /measurement/spec_run/adj_photo (avoids redundancy)
│       └── exposure             # Scalar (adj_photo_exposure)
│
├── cd/                          # Circular dichroism results
│   ├── wavelength               # 1D array
│   ├── front_CD                 # 1D array
│   ├── back_CD                  # 1D array
│   ├── front_abs                # 1D array
│   ├── back_abs                 # 1D array
│   ├── gen_CD                   # 1D array (genuine CD)
│   ├── ldlb_CD                  # 1D array (linear dichroism + birefringence)
│   ├── abs_avg                  # 1D array (average absorbance)
│   └── file_info/
│       ├── front_file           # String (source filename)
│       └── back_file            # String (source filename)
│
└── xrd/                         # X-ray diffraction results
    ├── 2theta_Cu                # 1D array (Cu K-alpha scale)
    ├── 2theta_Co                # 1D array (original Co K-alpha scale)
    ├── int                      # 1D array (intensity)
    └── file_info/
        └── source_file          # String (source filename)
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import List, Dict, Optional

# Import from existing modules
from h5_analysis import (
    filter_h5_files,
    read_h5_file,
    process_uv_vis_data_raw,
    process_pl_data_raw,
)


def find_associated_files(
    h5_file_path: str,
    base_dir: str,
    xrd_filename_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[str]]:
    """
    Find CD and XRD files associated with an H5 file.

    Args:
        h5_file_path: Path to the H5 file
        base_dir: Base directory containing subdirectories (CBox, CD, XRD)
        xrd_filename_mapping: Pre-computed mapping of sample[:13] to XRD filenames

    Returns:
        Dictionary with paths to associated files
    """
    # Extract sample[:13] from H5 filename according to notebook specification
    # Format: {batch_name}_{sample_position}_{sample[:13]}_run{n}_spec_run
    h5_path = Path(h5_file_path)
    filename = h5_path.stem

    # Split filename and parse according to the specification
    parts = filename.split("_")
    sample_id = None

    # Find the run part to locate sample[:13]
    for i, part in enumerate(parts):
        if (
            part.startswith("run") and i >= 2
        ):  # Need at least batch_name, sample_position, sample[:13] before run
            # The sample[:13] should be the part just before "run{n}"
            sample_id = parts[i - 1]
            break

    if not sample_id:
        print(f"Warning: Could not extract sample[:13] from filename: {filename}")
        return {"cd_front": None, "cd_back": None, "xrd": None}

    # Ensure we use only first 13 characters as specified in notebook
    sample_id = sample_id[:13]

    associated_files = {"cd_front": None, "cd_back": None, "xrd": None}

    base_path = Path(base_dir)

    # Search for CD files in {base_dir}/CD/**/*.txt
    # Pattern: front-{sample[:13]}.txt and back-{sample[:13]}.txt
    cd_dir = base_path / "CD"
    if cd_dir.exists():
        cd_front_pattern = f"front-{sample_id}.txt"
        cd_back_pattern = f"back-{sample_id}.txt"

        for cd_file in cd_dir.rglob(cd_front_pattern):
            associated_files["cd_front"] = str(cd_file)
            break

        for cd_file in cd_dir.rglob(cd_back_pattern):
            associated_files["cd_back"] = str(cd_file)
            break

    # Search for XRD files using pre-computed mapping
    # After remapping, XRD files follow {sample[:13]}.xy pattern
    if xrd_filename_mapping and sample_id in xrd_filename_mapping:
        xrd_file_path = xrd_filename_mapping[sample_id]
        if Path(xrd_file_path).exists():
            associated_files["xrd"] = xrd_file_path

    return associated_files


def create_xrd_filename_mapping(base_dir: str) -> Dict[str, str]:
    """
    Create a mapping of sample[:13] to XRD filenames using the same logic as xrd_plot_from_config.load_xy_data.

    Args:
        base_dir: Base directory containing XRD subdirectory

    Returns:
        Dictionary mapping sample[:13] to XRD file paths
    """
    base_path = Path(base_dir)
    xrd_dir = base_path / "XRD"
    mapping = {}

    if xrd_dir.exists():
        # Find all *_merge.xy files and sort them (same as xrd_plot_from_config.load_xy_data)
        xrd_files = list(xrd_dir.rglob("*_merge.xy"))
        if xrd_files:
            xrd_files_sorted = sorted(xrd_files, key=lambda f: f.name)

            # Map each file to its reindexed name (1.xy, 2.xy, etc.)
            for idx, xy_file in enumerate(xrd_files_sorted, 1):
                reindexed_name = f"{idx}"  # This becomes the sample ID for mapping
                mapping[reindexed_name] = str(xy_file)
                print(
                    f"XRD mapping: {xy_file.name} -> {reindexed_name}.xy -> {reindexed_name}"
                )

    return mapping


def process_cbox_data(h5_data: Dict) -> Dict:
    """
    Process CBox UV-Vis and PL data with individual and averaged spectra.

    Args:
        h5_data: Dictionary containing H5 file data

    Returns:
        Dictionary with processed CBox data
    """
    cbox_results = {}

    # Process UV-Vis data
    if "wl_data" in h5_data:
        try:
            print("Processing UV-Vis data...")
            uv_vis_results = process_uv_vis_data_raw(h5_data["wl_data"])
            print("UV-Vis data processed successfully")

            print(f"Wavelength axis shape: {uv_vis_results['wavelength'].shape}")
            print(
                f"Wavelength range: {uv_vis_results['wavelength'].min():.2f} - {uv_vis_results['wavelength'].max():.2f}"
            )
            print(f"Sample spectrum shape: {uv_vis_results['absorption'].shape}")
            print(
                f"UV-Vis results prepared with {uv_vis_results['num_spectra']} spectra"
            )

            cbox_results["uv_vis"] = {
                "wavelength": uv_vis_results["wavelength"],
                "energy": uv_vis_results["energy"],
                "absorption": uv_vis_results["absorption"],
                "absorption_avg": np.mean(uv_vis_results["absorption"], axis=0),
                "absorption_jacobian": uv_vis_results["absorption_jacobian"],
                "absorption_jacobian_avg": np.mean(
                    uv_vis_results["absorption_jacobian"], axis=0
                ),
                "transmission": uv_vis_results["transmission"],
                "transmission_avg": np.mean(uv_vis_results["transmission"], axis=0),
                "num_spectra": uv_vis_results["num_spectra"],
            }

        except Exception as e:
            print(f"Error processing UV-Vis data: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Process PL data
    if "pl_data" in h5_data:
        try:
            print("Processing PL data...")
            pl_results = process_pl_data_raw(h5_data["pl_data"])
            print("PL data processed successfully")

            print(f"PL spectra shape: {pl_results['intensity'].shape}")
            print(f"PL wavelength shape: {pl_results['wavelength'].shape}")
            print(f"PL results prepared with {pl_results['num_spectra']} spectra")

            cbox_results["pl"] = {
                "wavelength": pl_results["wavelength"],
                "energy": pl_results["energy"],
                "intensity": pl_results["intensity"],
                "intensity_avg": np.mean(pl_results["intensity"], axis=0),
                "intensity_jacobian": pl_results["intensity_jacobian"],
                "intensity_jacobian_avg": np.mean(
                    pl_results["intensity_jacobian"], axis=0
                ),
                "num_spectra": pl_results["num_spectra"],
            }

        except Exception as e:
            print(f"Error processing PL data: {e}")
            import traceback

            traceback.print_exc()
            # Continue without PL data
            pass

    # Process photo data - prepare metadata but don't copy large image data
    if "photo" in h5_data:
        photo_data = h5_data["photo"]
        cbox_results["photo"] = {
            # Don't copy image data here - will be softlinked in save function
            "exposure": photo_data["adj_photo_exposure"],
            # Store a flag to indicate we want to link the image
            "_link_image": True,
        }

    return cbox_results


def process_cd_data(cd_front_file: str, cd_back_file: str) -> Dict:
    """
    Process CD data from front and back measurement files.

    Args:
        cd_front_file: Path to front CD measurement file
        cd_back_file: Path to back CD measurement file

    Returns:
        Dictionary with processed CD data
    """
    # Load CD data files with correct format (4 columns: wavelength, CD, HT VOLTAGE, absorbance)
    # Skip the first 3 rows (header information)
    df_front = pd.read_csv(
        cd_front_file, sep="\t", skiprows=3, names=["wavelength", "CD", "HT", "abs"]
    )
    df_back = pd.read_csv(
        cd_back_file, sep="\t", skiprows=3, names=["wavelength", "CD", "HT", "abs"]
    )

    # Ensure same wavelength grid
    wavelength = df_front["wavelength"].values
    front_CD = df_front["CD"].values
    back_CD = df_back["CD"].values
    front_abs = df_front["abs"].values
    back_abs = df_back["abs"].values

    # Calculate derived quantities
    gen_CD = (front_CD - back_CD) / 2  # Genuine CD
    ldlb_CD = (front_CD + back_CD) / 2  # Linear dichroism + birefringence
    abs_avg = (front_abs + back_abs) / 2  # Average absorbance

    return {
        "wavelength": wavelength,
        "front_CD": front_CD,
        "back_CD": back_CD,
        "front_abs": front_abs,
        "back_abs": back_abs,
        "gen_CD": gen_CD,
        "ldlb_CD": ldlb_CD,
        "abs_avg": abs_avg,
        "file_info": {"front_file": cd_front_file, "back_file": cd_back_file},
    }


def process_xrd_data(xrd_file: str) -> Dict:
    """
    Process XRD data from XY file.

    Args:
        xrd_file: Path to XRD .xy file

    Returns:
        Dictionary with processed XRD data
    """
    # Load XRD data
    df = pd.read_csv(xrd_file, sep="\t", comment="#", names=["2theta_Co", "int"])

    # Convert Co K-alpha to Cu K-alpha
    Co_K_alpha = 1.7902
    Cu_K_alpha = 1.5406
    df["2theta_Cu"] = df["2theta_Co"] * Cu_K_alpha / Co_K_alpha

    # Filter 2theta range
    df_filtered = df[(df["2theta_Cu"] > 4) & (df["2theta_Cu"] < 50.5)]

    return {
        "2theta_Cu": df_filtered["2theta_Cu"].values,
        "2theta_Co": df_filtered["2theta_Co"].values,
        "int": df_filtered["int"].values,
        "file_info": {"source_file": xrd_file},
    }


def save_results_to_h5(
    h5_file_path: str,
    output_h5_path: str,
    cbox_data: Dict,
    cd_data: Optional[Dict],
    xrd_data: Optional[Dict],
):
    """
    Save processed results to a new H5 file in the 'results' group.

    Args:
        h5_file_path: Path to the original H5 file
        output_h5_path: Path where the processed H5 file will be saved
        cbox_data: Processed CBox data
        cd_data: Processed CD data (optional)
        xrd_data: Processed XRD data (optional)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_h5_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy original H5 file to output location
    import shutil

    shutil.copy2(h5_file_path, output_h5_path)

    # Open the copied file and add results
    with h5py.File(output_h5_path, "a") as hf:
        # Create or recreate results group
        if "results" in hf:
            del hf["results"]
        results_grp = hf.create_group("results")

        # Save CBox data
        cbox_grp = results_grp.create_group("cbox")

        # UV-Vis data
        if "uv_vis" in cbox_data:
            uv_vis_grp = cbox_grp.create_group("uv_vis")
            uv_vis = cbox_data["uv_vis"]
            for key, value in uv_vis.items():
                uv_vis_grp.create_dataset(key, data=value)

        # PL data
        if "pl" in cbox_data:
            pl_grp = cbox_grp.create_group("pl")
            pl = cbox_data["pl"]
            for key, value in pl.items():
                pl_grp.create_dataset(key, data=value)

        # Photo data - use softlinks to avoid redundancy
        if "photo" in cbox_data:
            photo_grp = cbox_grp.create_group("photo")
            photo = cbox_data["photo"]

            # Handle image data with softlinking
            if photo.get("_link_image", False):
                # Check if the original adj_photo exists in the file
                if "measurement/spec_run/adj_photo" in hf:
                    # Create a proper soft link instead of copying the data
                    photo_grp["image"] = h5py.SoftLink(
                        "/measurement/spec_run/adj_photo"
                    )
                    print(
                        "  Created softlink: /results/cbox/photo/image -> /measurement/spec_run/adj_photo"
                    )
                else:
                    print(
                        "  Warning: Original adj_photo not found, cannot create softlink"
                    )

            # Handle other photo data normally (exposure is small)
            for key, value in photo.items():
                if key not in ["_link_image"]:  # Skip internal flags
                    photo_grp.create_dataset(key, data=value)

        # Save CD data
        if cd_data:
            cd_grp = results_grp.create_group("cd")
            for key, value in cd_data.items():
                if key == "file_info":
                    info_grp = cd_grp.create_group("file_info")
                    for info_key, info_value in value.items():
                        info_grp.create_dataset(
                            info_key, data=info_value.encode("utf-8")
                        )
                else:
                    cd_grp.create_dataset(key, data=value)

        # Save XRD data
        if xrd_data:
            xrd_grp = results_grp.create_group("xrd")
            for key, value in xrd_data.items():
                if key == "file_info":
                    info_grp = xrd_grp.create_group("file_info")
                    for info_key, info_value in value.items():
                        info_grp.create_dataset(
                            info_key, data=info_value.encode("utf-8")
                        )
                else:
                    xrd_grp.create_dataset(key, data=value)


def process_single_h5_file(
    h5_file_path: str,
    base_dir: str,
    xrd_filename_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> bool:
    """
    Process a single H5 file and integrate associated CD and XRD data.

    Args:
        h5_file_path: Path to the H5 file
        base_dir: Base directory containing subdirectories (CBox, CD, XRD)
        xrd_filename_mapping: Pre-computed mapping of sample[:13] to XRD filenames
        verbose: Whether to print progress information

    Returns:
        True if processing was successful, False otherwise
    """
    try:
        if verbose:
            print(f"Processing {h5_file_path}")

        # Read H5 file
        print("Reading H5 file...")
        h5_data = read_h5_file(h5_file_path)
        if h5_data is None:
            print(f"Failed to read H5 file: {h5_file_path}")
            return False
        print("H5 file read successfully")

        # Create output path in preprocessed subdirectory
        h5_path = Path(h5_file_path)
        base_path = Path(base_dir)

        # Get relative path from base_dir and create output path
        try:
            rel_path = h5_path.relative_to(base_path)
            output_h5_path = base_path / "preprocessed" / rel_path
        except ValueError:
            # Fallback if relative path calculation fails
            output_h5_path = base_path / "preprocessed" / h5_path.name

        # Find associated files
        print("Finding associated files...")
        associated_files = find_associated_files(
            h5_file_path, base_dir, xrd_filename_mapping
        )
        if verbose:
            print(f"Associated files: {associated_files}")

        # Process CBox data
        print("Processing CBox data...")
        cbox_data = process_cbox_data(h5_data)
        print("CBox data processed successfully")

        # Process CD data if available
        cd_data = None
        if associated_files["cd_front"] and associated_files["cd_back"]:
            try:
                print("Processing CD data...")
                cd_data = process_cd_data(
                    associated_files["cd_front"], associated_files["cd_back"]
                )
                if verbose:
                    print("CD data processed successfully")
            except Exception as e:
                print(f"Failed to process CD data: {e}")

        # Process XRD data if available
        xrd_data = None
        if associated_files["xrd"]:
            try:
                print("Processing XRD data...")
                xrd_data = process_xrd_data(associated_files["xrd"])
                if verbose:
                    print("XRD data processed successfully")
            except Exception as e:
                print(f"Failed to process XRD data: {e}")

        # Save results to new H5 file in preprocessed directory
        print("Saving results...")
        save_results_to_h5(
            h5_file_path, str(output_h5_path), cbox_data, cd_data, xrd_data
        )

        if verbose:
            print(f"Results saved to {output_h5_path}")

        return True

    except Exception as e:
        print(f"Error processing {h5_file_path}: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_file_associations(
    base_dirs: List[str], verbose: bool = True
) -> Dict[str, Dict]:
    """
    Check file associations for all H5 files without processing them.

    Args:
        base_dirs: List of base directories containing CBox, CD, and XRD subdirectories
        verbose: Whether to print detailed output

    Returns:
        Dictionary with association results for each directory
    """
    results = {}

    for base_dir in base_dirs:
        print(f"\n=== Checking file associations for {base_dir} ===")

        # Create XRD filename mapping
        if verbose:
            print("Creating XRD filename mapping...")
        xrd_mapping = create_xrd_filename_mapping(base_dir)

        if verbose and xrd_mapping:
            print(f"Found {len(xrd_mapping)} XRD files:")
            for key, value in list(xrd_mapping.items())[:5]:  # Show first 5
                print(f"  {key} -> {Path(value).name}")
            if len(xrd_mapping) > 5:
                print(f"  ... and {len(xrd_mapping) - 5} more")

        # Find H5 files
        base_path = Path(base_dir)
        cbox_dir = base_path / "CBox"
        h5_files = []

        if cbox_dir.exists():
            h5_files = list(cbox_dir.rglob("*.h5"))
            print(f"Found {len(h5_files)} H5 files")
        else:
            print(f"Warning: CBox directory does not exist: {cbox_dir}")
            continue

        # Filter H5 files
        if h5_files:
            filtered_files_dict, ignored_files = filter_h5_files(
                [str(f) for f in h5_files]
            )
            filtered_h5_files = [
                info["file_path"] for info in filtered_files_dict.values()
            ]

            if ignored_files:
                print(
                    f"Ignored {len(ignored_files)} files: {[Path(f).name for f in ignored_files]}"
                )

            print(f"Processing {len(filtered_h5_files)} filtered H5 files")
        else:
            filtered_h5_files = []

        # Check associations for each H5 file
        dir_results = {
            "total_files": len(filtered_h5_files),
            "with_cd": 0,
            "with_xrd": 0,
            "with_both": 0,
            "with_neither": 0,
            "associations": [],
        }

        for i, h5_file in enumerate(filtered_h5_files):
            h5_name = Path(h5_file).name

            # Find associated files
            associated_files = find_associated_files(h5_file, base_dir, xrd_mapping)

            has_cd = associated_files["cd_front"] and associated_files["cd_back"]
            has_xrd = associated_files["xrd"] is not None

            if has_cd:
                dir_results["with_cd"] += 1
            if has_xrd:
                dir_results["with_xrd"] += 1
            if has_cd and has_xrd:
                dir_results["with_both"] += 1
            if not has_cd and not has_xrd:
                dir_results["with_neither"] += 1

            association_result = {
                "h5_file": h5_name,
                "cd_front": Path(associated_files["cd_front"]).name
                if associated_files["cd_front"]
                else None,
                "cd_back": Path(associated_files["cd_back"]).name
                if associated_files["cd_back"]
                else None,
                "xrd": Path(associated_files["xrd"]).name
                if associated_files["xrd"]
                else None,
                "has_cd": has_cd,
                "has_xrd": has_xrd,
            }
            dir_results["associations"].append(association_result)

            if verbose:
                status = []
                if has_cd:
                    status.append("CD")
                if has_xrd:
                    status.append("XRD")
                status_str = f"[{', '.join(status) if status else 'none'}]"
                print(f"  {i + 1:2d}. {h5_name} {status_str}")

                if verbose and (
                    associated_files["cd_front"]
                    or associated_files["cd_back"]
                    or associated_files["xrd"]
                ):
                    if associated_files["cd_front"]:
                        print(
                            f"      CD front: {Path(associated_files['cd_front']).name}"
                        )
                    if associated_files["cd_back"]:
                        print(
                            f"      CD back:  {Path(associated_files['cd_back']).name}"
                        )
                    if associated_files["xrd"]:
                        print(f"      XRD:      {Path(associated_files['xrd']).name}")

        # Summary for this directory
        total = dir_results["total_files"]
        print(f"\n--- Summary for {base_dir} ---")
        print(f"Total H5 files: {total}")
        print(
            f"With CD data:   {dir_results['with_cd']} ({dir_results['with_cd'] / total * 100:.1f}%)"
        )
        print(
            f"With XRD data:  {dir_results['with_xrd']} ({dir_results['with_xrd'] / total * 100:.1f}%)"
        )
        print(
            f"With both:      {dir_results['with_both']} ({dir_results['with_both'] / total * 100:.1f}%)"
        )
        print(
            f"With neither:   {dir_results['with_neither']} ({dir_results['with_neither'] / total * 100:.1f}%)"
        )

        results[base_dir] = dir_results

    return results


def process_h5_files_batch(
    base_dirs: List[str],
    verbose: bool = True,
    ignore_duplicates: bool = True,
    max_files: Optional[int] = None,
) -> List[str]:
    """
    Process all H5 files found in the specified base directories.

    Args:
        base_dirs: List of base directories containing CBox, CD, and XRD subdirectories
        verbose: Whether to print progress information
        ignore_duplicates: Whether to ignore duplicate H5 files
        max_files: Maximum number of files to process (for testing)

    Returns:
        List of successfully processed H5 files
    """
    # Create XRD filename mappings for each directory
    xrd_mappings = {}
    for base_dir in base_dirs:
        if verbose:
            print(f"Creating XRD filename mapping for {base_dir}...")
        xrd_mappings[base_dir] = create_xrd_filename_mapping(base_dir)

    # Find all H5 files in CBox subdirectories
    all_h5_files = []

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: Base directory does not exist: {base_dir}")
            continue

        cbox_dir = base_path / "CBox"
        if cbox_dir.exists():
            h5_files = list(cbox_dir.rglob("*.h5"))
            if verbose:
                print(f"Found {len(h5_files)} H5 files in {cbox_dir}")
            all_h5_files.extend([str(f) for f in h5_files])
        else:
            print(f"Warning: CBox directory does not exist: {cbox_dir}")

    if ignore_duplicates:
        # Remove duplicates while preserving order
        seen = set()
        unique_h5_files = []
        for f in all_h5_files:
            if f not in seen:
                seen.add(f)
                unique_h5_files.append(f)
        all_h5_files = unique_h5_files

    # Filter H5 files using existing filter function
    filtered_files_dict, ignored_files = filter_h5_files(all_h5_files)

    # Extract file paths from the filtered dictionary
    filtered_h5_files = [info["file_path"] for info in filtered_files_dict.values()]

    # Limit number of files if max_files is specified
    if max_files is not None and max_files > 0:
        original_count = len(filtered_h5_files)
        filtered_h5_files = filtered_h5_files[:max_files]
        if verbose:
            print(
                f"Limiting to first {len(filtered_h5_files)} files out of {original_count} total files"
            )

    if verbose:
        print(f"Found {len(filtered_h5_files)} H5 files to process")

    # Process each H5 file
    processed_files = []
    failed_files = []

    for h5_file in filtered_h5_files:
        print(f"Processing file: {h5_file} (type: {type(h5_file)})")

        # Determine which base directory this H5 file belongs to
        h5_path = Path(h5_file)
        base_dir = None

        for bd in base_dirs:
            bd_path = Path(bd)
            try:
                h5_path.relative_to(bd_path)
                base_dir = bd
                break
            except ValueError:
                continue

        if base_dir is None:
            print(f"Warning: Could not determine base directory for {h5_file}")
            failed_files.append(h5_file)
            continue

        print(f"Using base directory: {base_dir}")
        success = process_single_h5_file(
            h5_file, base_dir, xrd_mappings.get(base_dir), verbose=verbose
        )
        if success:
            processed_files.append(h5_file)
        else:
            failed_files.append(h5_file)

    if verbose:
        print("\nProcessing complete:")
        print(f"  Successfully processed: {len(processed_files)}")
        print(f"  Failed: {len(failed_files)}")
        if failed_files:
            print(f"  Failed files: {failed_files}")

    return processed_files


def main():
    """
    Main function to run the LPS preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="LPS Preprocessing Pipeline - Process H5 files and integrate CD/XRD data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Structure Expected:
  {dir_path}/
  ├── CBox/**/*.h5           # H5 files to process
  ├── CD/**/*.txt            # CD measurement files  
  ├── XRD/**/*_merge.xy      # XRD measurement files
  └── preprocessed/          # Output directory (created automatically)

Example usage:
  # Check file associations only (fast, good for debugging)
  uv run lps-preprocessing.py "G:\\My Drive\\LPS\\LPS-1" --check-only
  
  # Process all files  
  uv run lps-preprocessing.py "G:\\My Drive\\LPS\\LPS-1"
  
  # Process only first 3 files (useful for testing)
  uv run lps-preprocessing.py "G:\\My Drive\\LPS\\LPS-1" --max-files 3
  
  # Process multiple directories
  uv run lps-preprocessing.py "G:\\My Drive\\LPS\\20250709_S_MeOMBAI_prestudy_1" "G:\\My Drive\\LPS\\20250709_S_MeOMBAI_prestudy_2"
        """,
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more base directories containing CBox, CD, and XRD subdirectories",
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
        "--allow-duplicates",
        action="store_true",
        help="Allow processing of duplicate H5 files (default: ignore duplicates)",
    )

    parser.add_argument(
        "--check-only",
        "-c",
        action="store_true",
        help="Only check file associations without processing (useful for debugging and prerequisite testing)",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit processing to first N files (useful for testing). Only applies when not using --check-only",
    )

    args = parser.parse_args()

    # Handle verbose flag
    verbose = args.verbose and not args.quiet
    ignore_duplicates = not args.allow_duplicates

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
        print("Expected structure for each directory:")
        print("  CBox/**/*.h5           # H5 files to process")
        print("  CD/**/*.txt            # CD measurement files")
        print("  XRD/**/*_merge.xy      # XRD measurement files")
        print("  preprocessed/          # Output directory for processed H5 files")
        print()

    # Process H5 files or just check associations
    try:
        if args.check_only:
            print("=== FILE ASSOCIATION CHECK MODE ===")
            print("Checking file associations without processing...")
            print()

            check_results = check_file_associations(valid_dirs, verbose=verbose)

            # Overall summary
            total_files = sum(
                result["total_files"] for result in check_results.values()
            )
            total_with_cd = sum(result["with_cd"] for result in check_results.values())
            total_with_xrd = sum(
                result["with_xrd"] for result in check_results.values()
            )
            total_with_both = sum(
                result["with_both"] for result in check_results.values()
            )
            total_with_neither = sum(
                result["with_neither"] for result in check_results.values()
            )

            print("\n=== OVERALL SUMMARY ===")
            print(f"Total H5 files across all directories: {total_files}")
            print(
                f"With CD data:   {total_with_cd} ({total_with_cd / total_files * 100:.1f}%)"
            )
            print(
                f"With XRD data:  {total_with_xrd} ({total_with_xrd / total_files * 100:.1f}%)"
            )
            print(
                f"With both:      {total_with_both} ({total_with_both / total_files * 100:.1f}%)"
            )
            print(
                f"With neither:   {total_with_neither} ({total_with_neither / total_files * 100:.1f}%)"
            )

            if total_with_neither > 0:
                print(
                    f"\nNote: {total_with_neither} files have no associated CD or XRD data."
                )
                print(
                    "This is normal if not all samples have corresponding measurements."
                )

            if total_files > 0:
                print("\nFile association check completed successfully!")
                print("You can now run without --check-only to process the files.")
            else:
                print("No H5 files found to check.")

        else:
            processed_files = process_h5_files_batch(
                valid_dirs,
                verbose=verbose,
                ignore_duplicates=ignore_duplicates,
                max_files=args.max_files,
            )

            if verbose:
                print("\nFinal Summary:")
            print(f"Successfully processed {len(processed_files)} H5 files")

            if not processed_files:
                print("No files were processed. Please check:")
                print("1. H5 files exist in CBox subdirectories")
                print("2. H5 files match the expected naming format")
                print("3. Directory structure matches expected layout")
                print("4. Run with --check-only flag to verify file associations")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

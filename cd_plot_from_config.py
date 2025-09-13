import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
from pathlib import Path
from plot_helper import (
    get_linestyle_factory,
    get_color_factory,
    get_label_from_mapping,
    match_label_filter,
)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_cd_files(data_dirs, file_pattern="*.txt"):
    """Find CD files matching the specified pattern."""
    files = []
    for d in data_dirs:
        files.extend(Path(d).glob(file_pattern))
    return sorted(files)


def rename_file_if_needed(filename, rename_rules=None):
    """Apply filename renaming rules if specified in config."""
    if not rename_rules:
        return filename

    stem = filename.stem
    for rule in rename_rules:
        if rule["pattern"] in stem:
            parts = stem.split(rule["pattern"])
            base = parts[0]
            num = int(parts[1])
            new_num = num + rule["offset"]
            new_stem = f"{base}{rule['new_pattern']}{new_num}"
            new_filename = filename.with_stem(new_stem)

            if new_filename.exists():
                print(
                    f"File '{new_filename.name}' already exists. Using existing file."
                )
                return new_filename

            shutil.copy(filename, new_filename)
            print(f"Copied '{filename.name}' to '{new_filename.name}'")
            return new_filename

    return filename


def load_cd_data(txt_files, label_mapping, file_params, rename_rules=None):
    """Load and process CD data from text files."""
    dfs = []
    common_cols = set()

    for txt_file in txt_files:
        # Apply renaming if needed
        txt_file = rename_file_if_needed(txt_file, rename_rules)

        # Determine column naming based on front/back in filename
        if "front" in txt_file.stem:
            get_col = lambda col: f"front_{col}_{txt_file.stem.replace('front', '')}"
        elif "back" in txt_file.stem:
            get_col = lambda col: f"back_{col}_{txt_file.stem.replace('back', '')}"
        else:
            print(
                f"Skipping file {txt_file} - does not contain 'front' or 'back' in its name."
            )
            continue

        # Track common column identifiers
        common_cols.add(txt_file.stem.replace("front", "").replace("back", ""))

        # Read the data file
        df = pd.read_csv(
            txt_file,
            sep=file_params.get("separator", "\t"),
            names=["wavelength", get_col("CD"), get_col("HT"), get_col("abs")],
            skiprows=file_params.get("skiprows", 3),
        )
        df = df.set_index("wavelength")

        # Apply wavelength filters if specified
        if "wavelength_range" in file_params:
            wl_min, wl_max = file_params["wavelength_range"]
            if wl_min is not None:
                df = df[df.index >= wl_min]
            if wl_max is not None:
                df = df[df.index <= wl_max]

        dfs.append(df)

    # Combine all dataframes
    df = pd.concat(dfs, axis=1)

    # Calculate derived quantities for each sample
    for common_col in sorted(common_cols):
        front_cd_col = f"front_CD_{common_col}"
        back_cd_col = f"back_CD_{common_col}"
        front_abs_col = f"front_abs_{common_col}"
        back_abs_col = f"back_abs_{common_col}"

        # Check if both front and back columns exist
        if front_cd_col in df.columns and back_cd_col in df.columns:
            # Genuine CD (average of front and back)
            df[f"gen_CD_{common_col}"] = (df[front_cd_col] + df[back_cd_col]) / 2

            # Linear dichroism + linear birefringence (LDLB)
            df[f"ldlb_CD_{common_col}"] = df[front_cd_col] - df[back_cd_col]

        if front_abs_col in df.columns and back_abs_col in df.columns:
            # Average absorption
            df[f"abs_{common_col}"] = (df[front_abs_col] + df[back_abs_col]) / 2

            # g-factor calculation (if genuine CD exists)
            if f"gen_CD_{common_col}" in df.columns:
                df[f"g_factor_{common_col}"] = df[f"gen_CD_{common_col}"] / (
                    df[f"abs_{common_col}"] * 32980
                )

    # Create sample labels using the mapping
    sample_labels = {}
    for common_col in common_cols:
        # Extract sample number from common_col (assumes format ends with -number)
        sample_number = common_col.split("-")[-1] if "-" in common_col else common_col
        label = get_label_from_mapping(sample_number, label_mapping)
        sample_labels[common_col] = label

    return df, common_cols, sample_labels


def plot_single_column_data(
    ax,
    df,
    column_prefix,
    sample_labels,
    filtered_common_cols=None,
    xlabel="wavelength (nm)",
    ylabel="",
    scientific_notation=False,
    lw=0.8,
    get_linestyle=None,
    get_color=None,
    legend_kwargs=None,
):
    """Plot single column data with consistent styling."""
    if legend_kwargs is None:
        legend_kwargs = {}

    if get_linestyle is None:
        get_linestyle = lambda label: "-"

    if get_color is None:
        get_color = lambda label: None

    # Get label function that maps column names to display labels
    def get_label_func(col):
        # Extract common_col from column name
        common_col = col.replace(column_prefix, "")
        return sample_labels.get(common_col, common_col)

    # Sort columns by their display labels, filtering by common_cols if provided
    if filtered_common_cols is not None:
        # Only include columns for filtered samples
        columns = [
            f"{column_prefix}{col}"
            for col in filtered_common_cols
            if f"{column_prefix}{col}" in df.columns
        ]
    else:
        columns = [c for c in df.columns if c.startswith(column_prefix)]
    columns = sorted(columns, key=get_label_func)

    for col in columns:
        df_to_plot = df[[col]].dropna()
        if len(df_to_plot) == 0:
            continue

        label = get_label_func(col)
        linestyle = get_linestyle(label)
        color = get_color(label)

        ax.plot(
            df_to_plot.index,
            df_to_plot[col],
            label=label,
            lw=lw,
            linestyle=linestyle,
            color=color,
        )

    ax.legend(**legend_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if scientific_notation:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)


def plot_front_back_pairs(
    ax,
    df,
    common_cols,
    sample_labels,
    data_type,
    xlabel="wavelength (nm)",
    ylabel="",
    lw=0.8,
    get_linestyle=None,
    get_color=None,
    legend_kwargs=None,
):
    """Plot front/back paired data with consistent styling and legend."""
    if legend_kwargs is None:
        legend_kwargs = {}

    def default_linestyle_func(label):
        """Default linestyle function: solid for front, dotted for back."""
        return "-" if label.endswith("_front") else ":"

    if get_linestyle is None:
        get_linestyle = default_linestyle_func

    if get_color is None:
        get_color = lambda label: None

    for col in sorted(common_cols, key=lambda x: sample_labels.get(x, x)):
        front_col = f"front_{data_type}_{col}"
        back_col = f"back_{data_type}_{col}"

        try:
            front_df_to_plot = df[[front_col]].dropna()
            back_df_to_plot = df[[back_col]].dropna()
        except KeyError:
            print(f"Skipping {col} due to missing front/back columns.")
            continue

        if len(front_df_to_plot) == 0 or len(back_df_to_plot) == 0:
            continue

        label = sample_labels.get(col, col)

        # Use the same base label for both front and back to avoid doubling linestyle assignments
        base_linestyle = get_linestyle(label)
        base_color = get_color(label)

        # Front line - solid style
        front_line = ax.plot(
            front_df_to_plot.index,
            front_df_to_plot[front_col],
            label=f"{label}_front",
            lw=lw,
            linestyle=base_linestyle if base_linestyle else "-",
            color=base_color,
        )

        # Back line - dotted style with same color
        ax.plot(
            back_df_to_plot.index,
            back_df_to_plot[back_col],
            label=f"{label}_back",
            lw=lw,
            linestyle=":",  # Always use dotted for back
            color=base_color if base_color else front_line[0].get_color(),
        )

    # Configure legend with front/back grouping showing both line styles
    handles, labels = ax.get_legend_handles_labels()

    # Group front/back pairs and create custom legend entries
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    grouped_handles = []
    grouped_labels = []

    # Get unique base labels (samples)
    base_labels = set()
    for label in labels:
        base_label = label.replace("_front", "").replace("_back", "")
        base_labels.add(base_label)

    # Create grouped legend entries
    for base_label in sorted(base_labels):
        front_label = f"{base_label}_front"
        back_label = f"{base_label}_back"

        # Find handles for front and back
        front_idx = labels.index(front_label) if front_label in labels else None
        back_idx = labels.index(back_label) if back_label in labels else None

        if front_idx is not None and back_idx is not None:
            # Create two separate line artists that will share the same label
            front_handle = handles[front_idx]
            back_handle = handles[back_idx]

            # Create solid line for front
            solid_line = Line2D(
                [0],
                [0],
                color=front_handle.get_color(),
                linestyle="-",
                linewidth=front_handle.get_linewidth(),
            )

            # Create dotted line for back
            dotted_line = Line2D(
                [0],
                [0],
                color=back_handle.get_color(),
                linestyle=":",
                linewidth=back_handle.get_linewidth(),
            )

            # Add both handles as a tuple for the same label
            grouped_handles.append((solid_line, dotted_line))
            grouped_labels.append(f"{base_label} front/back")

    # Set default ncols if not specified
    if "ncols" not in legend_kwargs:
        legend_kwargs["ncols"] = 1  # Single column for cleaner look with longer labels

    # Remove fontsize from legend_kwargs to avoid duplication
    legend_kwargs_copy = legend_kwargs.copy()
    legend_kwargs_copy.pop("fontsize", None)

    # Create legend with custom handler for tuples
    legend = ax.legend(
        handles=grouped_handles,
        labels=grouped_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        framealpha=0,
        fontsize="small",
        **legend_kwargs_copy,
    )
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.1))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_cd_data(df, common_cols, sample_labels, plot_cfg, data_dir=None):
    """Generate CD plots based on configuration."""
    print(f"Generating plot: {plot_cfg['output_file']}")

    # Filter samples if specified
    if "filters" in plot_cfg:
        filtered_cols = filter_columns_by_labels(sample_labels, plot_cfg["filters"])
        # Filter common_cols to only include those that match the filters
        common_cols = [col for col in common_cols if col in filtered_cols]
        if not common_cols:
            print(
                f"  Skipping plot {plot_cfg['output_file']} as no data matched the filters."
            )
            return

    # Create styling functions from config using plot_helper factories
    style_config = plot_cfg.get("style", {})

    linecolor_config = style_config.get("linecolor", {})
    get_color = get_color_factory(**linecolor_config)

    # Create separate color factory for front/back plots to avoid same-prefix confusion
    front_back_linecolor_config = linecolor_config.copy()
    if front_back_linecolor_config.get("cycle_by") == "prefix":
        front_back_linecolor_config["cycle_by"] = "line"
    get_color_front_back = get_color_factory(**front_back_linecolor_config)

    linestyle_config = style_config.get("linestyle", {})
    get_linestyle = get_linestyle_factory(**linestyle_config)

    # Create figure with 6 subplots (3 rows x 2 columns) with shared x-axis for each column
    fig, ((ax_cd_gen, ax_ldlb), (ax_g_factor, ax_cd), (ax_uv_vis, ax_uv_vis_fb)) = (
        plt.subplots(nrows=3, ncols=2, figsize=(13, 10), dpi=300, sharex="col")
    )

    if plot_cfg.get("title"):
        fig.suptitle(plot_cfg["title"], fontsize=16)

    # Plot options
    legend_kwargs = plot_cfg.get("legend_kwargs", {})

    # Plot single-column data
    plot_single_column_data(
        ax_cd_gen,
        df,
        "gen_CD_",
        sample_labels,
        filtered_common_cols=common_cols,
        ylabel="$CD_{gen}$ (mdeg)",
        get_linestyle=get_linestyle,
        get_color=get_color,
        legend_kwargs=legend_kwargs.copy(),
    )

    plot_single_column_data(
        ax_ldlb,
        df,
        "ldlb_CD_",
        sample_labels,
        filtered_common_cols=common_cols,
        ylabel="LDLB CD (mdeg)",
        get_linestyle=get_linestyle,
        get_color=get_color,
        legend_kwargs={**legend_kwargs},
    )

    plot_single_column_data(
        ax_uv_vis,
        df,
        "abs_",
        sample_labels,
        filtered_common_cols=common_cols,
        ylabel="abs (a.u.)",
        get_linestyle=get_linestyle,
        get_color=get_color,
        legend_kwargs=legend_kwargs.copy(),
    )

    # Dynamic scaling for g-factor plotting
    df_g_scaled = df.copy()
    g_factor_cols = [col for col in df.columns if col.startswith("g_factor_")]

    if g_factor_cols:
        # Find the maximum absolute g-factor value across all columns
        all_g_values = []
        for col in g_factor_cols:
            values = df[col].dropna()
            if len(values) > 0:
                all_g_values.extend(values.abs())

        max_g = max(all_g_values) if all_g_values else 0

        if max_g > 0:
            # Determine scaling based on max value order of magnitude
            order_of_magnitude = int(np.floor(np.log10(max_g)))

            # Scale to bring max value to range 1-100
            if order_of_magnitude >= 0:  # Values >= 1
                scale_factor = 1
                scale_label = ""
            else:  # Values < 1, scale up
                scale_exponent = -order_of_magnitude + 1
                scale_factor = 10**scale_exponent
                scale_label = f" ($\\times$10$^{{-{scale_exponent}}}$)"

            # Apply scaling
            for col in g_factor_cols:
                df_g_scaled[col] = df_g_scaled[col] * scale_factor

            ylabel_g = f"g factor{scale_label}"
        else:
            ylabel_g = "g factor"
    else:
        ylabel_g = "g factor"

    plot_single_column_data(
        ax_g_factor,
        df_g_scaled,
        "g_factor_",
        sample_labels,
        filtered_common_cols=common_cols,
        ylabel=ylabel_g,
        scientific_notation=False,
        get_linestyle=get_linestyle,
        get_color=get_color,
        legend_kwargs=legend_kwargs.copy(),
    )

    # Plot front/back paired data
    front_back_legend_kwargs = {**legend_kwargs, "ncols": 1, "loc": "lower left"}

    plot_front_back_pairs(
        ax_cd,
        df,
        common_cols,
        sample_labels,
        "CD",
        ylabel="$CD_{front/back}$ (mdeg)",
        get_linestyle=get_linestyle,
        get_color=get_color_front_back,
        legend_kwargs=front_back_legend_kwargs,
    )

    plot_front_back_pairs(
        ax_uv_vis_fb,
        df,
        common_cols,
        sample_labels,
        "abs",
        ylabel="abs (a.u.)",
        get_linestyle=get_linestyle,
        get_color=get_color_front_back,
        legend_kwargs=front_back_legend_kwargs,
    )

    # Hide x-axis labels for top and middle rows (only show on bottom row)
    for ax in [ax_cd_gen, ax_g_factor, ax_ldlb, ax_cd]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    # Set x-axis labels only for bottom row
    ax_uv_vis.set_xlabel("wavelength (nm)")
    ax_uv_vis_fb.set_xlabel("wavelength (nm)")

    # Final layout and save - remove vertical spacing between subplots in same column
    fig.tight_layout(pad=1.0, w_pad=0.3, h_pad=0.0)
    plt.subplots_adjust(hspace=0)

    # Construct output path
    if data_dir:
        output_path = Path(data_dir) / plot_cfg["output_file"]
    else:
        output_path = Path(plot_cfg["output_file"])

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Plot saved to: {output_path}")
    plt.close(fig)


def filter_columns_by_labels(sample_labels, filters):
    """Filter sample columns based on label criteria using plot_helper.match_label_filter."""
    if not filters:
        return list(sample_labels.keys())

    result = []
    for col, label in sample_labels.items():
        # Check if this label matches any of the filters
        if any(match_label_filter(label, filt) for filt in filters):
            result.append(col)

    return result


def main(config_path):
    """Main function to generate CD plots from text files based on a YAML config."""
    config = load_config(config_path)

    # Find CD files
    txt_files = find_cd_files(config["data_dirs"], config.get("file_pattern", "*.txt"))
    print(f"Found {len(txt_files)} CD files.")

    # Load and process data
    df, common_cols, sample_labels = load_cd_data(
        txt_files,
        config["label_mapping"],
        config.get("file_params", {}),
        config.get("rename_rules", None),
    )

    print(f"Processed {len(common_cols)} samples: {list(sample_labels.values())}")

    # Use the first data directory for saving plots
    data_dir = config["data_dirs"][0] if config["data_dirs"] else None

    # Generate plots
    for plot_cfg in config["plots"]:
        plot_cd_data(df, common_cols, sample_labels, plot_cfg, data_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CD plots from text files based on a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cd_plot_config.yaml",
        help="Path to the YAML configuration file (default: cd_plot_config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)

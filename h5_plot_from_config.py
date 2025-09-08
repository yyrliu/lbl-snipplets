import argparse
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import the h5_analysis module containing all core functions
from h5_analysis import (
    discover_h5_files,
    read_h5_file,
    process_uv_vis_data,
    process_pl_data,
    wavelength_to_energy,
    apply_jacobian,
)
from plot_helper import get_color_factory, get_linestyle_factory


def get_label_from_mapping(metadata, label_mapping):
    """Extract a clean label from sample ID using regex-based mapping from config with number_range and offset."""
    import re

    sample_name_full = str(metadata.get("sample_name", ""))
    default_pattern = "^(?P<prefix>[A-Za-z0-9]*?)[-_]?(?P<number>\d+)$"
    for mapping in label_mapping:
        mapping_pattern = mapping.get("pattern", default_pattern)
        for rule in mapping.get("rules", []):
            pattern = rule.get("pattern", mapping_pattern)
            search = re.search(pattern, sample_name_full)
            if search:
                prefix = search.group("prefix") or ""
                number = int(search.group("number"))
                if prefix == rule.get("prefix", ""):
                    range_str = rule.get("number_range", "")
                    if range_str:
                        try:
                            min_num, max_num = map(int, range_str.split("-"))
                        except Exception:
                            continue
                        if min_num <= number <= max_num:
                            number_mapping = rule.get("number_mapping", None)
                            offset_idx = number - min_num + 1
                            if number_mapping:
                                mapped_result = number_mapping.get(
                                    offset_idx, offset_idx
                                )
                                label = rule["label"].format(
                                    mapped_result=mapped_result
                                )
                            else:
                                display_number = number if min_num == 1 else offset_idx
                                label = rule["label"].format(number=display_number)
                            return label
    return f"Sample {sample_name_full}"


def plot_single_column_data(
    ax,
    dfs,
    x_col,
    y_col,
    xlabel="",
    ylabel="",
    scientific_notation=False,
    lw=None,
    get_linestyle_func=None,
    get_color_func=None,
):
    """Plot single column data with consistent styling."""
    for df in dfs:
        if x_col in df.columns and y_col in df.columns:
            plot_df = df[[x_col, y_col]].dropna()
            if len(plot_df) > 0:
                label = df["metadata"][0]["label"]
                linestyle = get_linestyle_func(label) if get_linestyle_func else "-"
                color = get_color_func(label) if get_color_func else None
                ax.plot(
                    plot_df[x_col],
                    plot_df[y_col],
                    label=label,
                    lw=lw,
                    linestyle=linestyle,
                    color=color,
                )

    handles, labels = ax.get_legend_handles_labels()

    # Create a unique legend
    by_label = dict(zip(labels, handles))
    sorted_by_label = sorted(by_label.items(), key=lambda x: x[0])

    if sorted_by_label:
        sorted_labels, sorted_handles = zip(*sorted_by_label)
        ax.legend(sorted_handles, sorted_labels, frameon=False, fontsize="small")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if scientific_notation:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)


def get_style_func(style_mapping, style_key, default_style):
    """Create a function that returns a style based on a label."""

    def func(label):
        if style_mapping:
            for mapping in style_mapping:
                if mapping["label_contains"] in label:
                    return mapping[style_key]
        return default_style

    return func


def main(config_path):
    """Main function to generate plots from H5 files based on a YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Discover H5 files
    h5_files = discover_h5_files(config["data_dirs"])
    print(f"Found {len(h5_files)} H5 files.")

    # Process H5 files and extract data
    uv_vis_dfs = []
    pl_dfs = []
    for h5_file in h5_files:
        data = read_h5_file(h5_file)
        if data is None:
            continue

        metadata = {
            **data["metadata"],
            "label": get_label_from_mapping(data["metadata"], config["label_mapping"]),
        }
        print(f"Mapped file {h5_file.name} to label: {metadata['label']}")
        metadata_df = pd.DataFrame([{"metadata": metadata}])

        # Process UV-Vis and PL data
        uv_vis_results = process_uv_vis_data(data["wl_data"], (350, 700))
        pl_results = process_pl_data(data["pl_data"], (350, 700))

        # Create UV-Vis dataframe
        uv_vis_df = pd.DataFrame(
            {
                "wavelength": uv_vis_results["wavelength"],
                "energy": uv_vis_results["energy"],
                "absorption": uv_vis_results["absorption"],
                "absorption_jacobian": uv_vis_results["absorption_jacobian"],
                "transmission": uv_vis_results["transmission"],
            }
        )
        uv_vis_df = pd.concat([uv_vis_df, metadata_df], axis=1)
        uv_vis_dfs.append(uv_vis_df)

        # Create PL dataframe
        if pl_results["average_spectrum"] is not None:
            avg_spec = pl_results["average_spectrum"]
            pl_df = pd.DataFrame(
                {
                    "wavelength": avg_spec["wavelength"],
                    "energy": wavelength_to_energy(avg_spec["wavelength"]),
                    "intensity": avg_spec["intensity"],
                    "intensity_jacobian": apply_jacobian(
                        avg_spec["intensity"], avg_spec["wavelength"]
                    ),
                }
            )
            pl_df = pd.concat([pl_df, metadata_df], axis=1)
            pl_dfs.append(pl_df)

    print(
        f"Successfully processed {len(uv_vis_dfs)} files with UV-Vis data and {len(pl_dfs)} files with PL data."
    )

    # Generate plots based on the configuration
    for plot_config in config["plots"]:
        print(f"Generating plot: {plot_config['output_file']}")

        # Filter data for the current plot
        def match_filter(label, filt):
            if "equals" in filt:
                return (
                    label.startswith(filt["equals"])
                    and label[len(filt["equals"]) :].strip().isdigit()
                )
            elif "contains" in filt:
                return filt["contains"] in label
            return False

        filtered_uv_vis_dfs = [
            df
            for df in uv_vis_dfs
            if any(
                match_filter(df["metadata"][0]["label"], filt)
                for filt in plot_config["filters"]
            )
        ]
        filtered_pl_dfs = [
            df
            for df in pl_dfs
            if any(
                match_filter(df["metadata"][0]["label"], filt)
                for filt in plot_config["filters"]
            )
        ]

        filtered_uv_vis_dfs = sorted(
            filtered_uv_vis_dfs, key=lambda df: df["metadata"][0]["label"]
        )
        filtered_pl_dfs = sorted(
            filtered_pl_dfs, key=lambda df: df["metadata"][0]["label"]
        )

        if not filtered_uv_vis_dfs and not filtered_pl_dfs:
            print(
                f"  Skipping plot {plot_config['output_file']} as no data matched the filters."
            )
            continue

        # Create styling functions from config using plot_helper factories
        style_config = plot_config.get("style", {})

        def get_style_funcs():
            get_color_func = get_color_factory(**style_config.get("linecolor", {}))
            get_linestyle_func = get_linestyle_factory(
                **style_config.get("linestyle", {})
            )
            return get_color_func, get_linestyle_func

        # Create figure
        fig, ((ax_uv_abs_wl, ax_uv_abs_energy), (ax_pl_int_wl, ax_pl_int_energy)) = (
            plt.subplots(nrows=2, ncols=2, figsize=(13, 9), dpi=300)
        )
        if plot_config.get("title"):
            fig.suptitle(plot_config["title"], fontsize=16)

        # Plot UV-Vis data
        get_color_func, get_linestyle_func = get_style_funcs()
        plot_single_column_data(
            ax_uv_abs_wl,
            filtered_uv_vis_dfs,
            "wavelength",
            "absorption",
            "Wavelength (nm)",
            "Abs",
            get_linestyle_func=get_linestyle_func,
            get_color_func=get_color_func,
        )
        # get_color_func, get_linestyle_func = get_style_funcs()
        plot_single_column_data(
            ax_uv_abs_energy,
            filtered_uv_vis_dfs,
            "energy",
            "absorption_jacobian",
            "Energy (eV)",
            "Abs (Jacobian)",
            get_linestyle_func=get_linestyle_func,
            get_color_func=get_color_func,
        )

        # Plot PL data
        # get_color_func, get_linestyle_func = get_style_funcs()
        plot_single_column_data(
            ax_pl_int_wl,
            filtered_pl_dfs,
            "wavelength",
            "intensity",
            "Wavelength (nm)",
            "PL Intensity",
            get_linestyle_func=get_linestyle_func,
            get_color_func=get_color_func,
        )
        # get_color_func, get_linestyle_func = get_style_funcs()
        plot_single_column_data(
            ax_pl_int_energy,
            filtered_pl_dfs,
            "energy",
            "intensity_jacobian",
            "Energy (eV)",
            "PL Intensity (Jacobian)",
            get_linestyle_func=get_linestyle_func,
            get_color_func=get_color_func,
        )

        # Final layout and save
        fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5, rect=[0, 0, 1, 0.96])
        output_path = Path(config["data_dirs"][0]) / plot_config["output_file"]
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Plot saved to: {output_path}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots for H5 files based on a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)

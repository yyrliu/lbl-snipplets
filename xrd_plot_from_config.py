import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from plot_helper import (
    get_linestyle_factory,
    get_color_factory,
    get_label_from_mapping,
    filter_columns,
    add_peak_label,
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_xy_files(data_dirs):
    files = []
    for d in data_dirs:
        files.extend(Path(d).glob("**/*.xy"))
    return files


def load_xy_data(xy_files, label_mapping):
    dfs = []
    labels = []
    # Sort files by name and reindex as 1.xy, 2.xy, ...
    xy_files_sorted = sorted(xy_files, key=lambda f: f.name)
    for idx, xy_file in enumerate(xy_files_sorted, 1):
        reindexed_name = f"{idx}.xy"
        label = get_label_from_mapping(str(idx), label_mapping)
        print(f"{xy_file.name} -> {reindexed_name} | label: {label}")
        df = pd.read_csv(xy_file, sep="\t", comment="#", names=["2theta_Co", "int"])
        # Convert to Cu K_alpha
        Co_K_alpha = 1.7902
        Cu_K_alpha = 1.5406
        df["2theta_Cu"] = df["2theta_Co"] * Cu_K_alpha / Co_K_alpha
        df = df[(df["2theta_Cu"] > 4) & (df["2theta_Cu"] < 50.5)]
        df = df.set_index("2theta_Cu")
        dfs.append(df["int"])
        labels.append(label)
    df_all = pd.concat(dfs, axis=1)
    df_all.columns = labels
    return df_all


def plot_xrd(df, plot_cfg, data_dir=None):
    cols = filter_columns(df, plot_cfg.get("filters", []))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 5), dpi=300, layout="tight")

    # Create styling functions from config using plot_helper factories
    style_config = plot_cfg.get("style", {})
    get_color = get_color_factory(**style_config.get("linecolor", {}))
    get_linestyle = get_linestyle_factory(**style_config.get("linestyle", {}))

    for ax in axs:
        for col in sorted(cols):
            df_to_plot = df[[col]].dropna()
            ax.plot(
                df_to_plot.index,
                df_to_plot[col],
                label=col,
                linestyle=get_linestyle(col),
                color=get_color(col),
            )
        ax.set_yscale("log")
        ax.set_xlabel("$2\\theta$ (degrees, Cu K$\\alpha$)")
        ax.set_ylabel("intensity (a.u.)")
        ax.set_ylim(*plot_cfg["options"]["ylim"])
        ax.legend()
        ax.minorticks_on()
        from matplotlib.ticker import AutoMinorLocator

        minor_locator = AutoMinorLocator(10)
        ax.xaxis.set_minor_locator(minor_locator)

        # Add peak labels using the smart positioning function
        for peak in plot_cfg.get("peaks", []):
            # Use the filtered data for peak positioning
            filtered_df = df[cols]
            add_peak_label(
                ax,
                filtered_df,
                peak["position"],
                peak["label"],
                x_range=peak.get("x_range", 1.0),
                y_offset=peak.get("y_offset", None),
                fontdict=peak.get("fontdict", None),
            )

    axs[0].set_xlim(*plot_cfg["options"]["xlim"])
    if "xlim_2" in plot_cfg["options"]:
        axs[1].set_xlim(*plot_cfg["options"]["xlim_2"])

    # Construct output path - save to data directory if provided, otherwise use current directory
    if data_dir:
        output_path = Path(data_dir) / plot_cfg["output_file"]
    else:
        output_path = Path(plot_cfg["output_file"])

    fig.savefig(output_path)
    print(f"Saved plot to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    xy_files = find_xy_files(cfg["data_dirs"])
    df = load_xy_data(xy_files, cfg["label_mapping"])

    # Use the first data directory for saving plots
    data_dir = cfg["data_dirs"][0] if cfg["data_dirs"] else None

    for plot_cfg in cfg["plots"]:
        plot_xrd(df, plot_cfg, data_dir)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
from numpy import linspace


def get_label_prefix(label):
    # Extract prefix from label (everything before the first space or underscore)
    if " " in label:
        return label.split(" ")[0]
    elif "_" in label:
        return label.split("_")[0]
    else:
        return label


def get_linestyle_factory(
    cycle_within=None, ignore_out_of_range=False, linestyles=["-", "--", ":", "-."]
):
    # Create a linestyle mapping using closure with dict for prefix tracking

    if cycle_within is None:
        return lambda label: None

    if cycle_within == "prefix":
        linestyle_iters = {}
        linestyle_mapping = {}

        def get_linestyle(label):
            prefix = get_label_prefix(label)

            # If prefix not seen before, assign new linestyle
            if prefix not in linestyle_iters.keys():
                linestyle_iters[prefix] = iter(linestyles)

            if label not in linestyle_mapping.keys():
                try:
                    linestyle_mapping[label] = next(linestyle_iters[prefix])
                except StopIteration:
                    if ignore_out_of_range:
                        linestyle_iters[prefix] = iter(linestyles)
                        return next(linestyle_iters[prefix])
                    else:
                        raise ValueError(f"Ran out of linestyles for prefix: {prefix}")

            return linestyle_mapping[label]

        return get_linestyle

    raise ValueError(f"Unknown cycle_within policy: {cycle_within}")


def get_color_factory(colormap="tab10", n=None, cycle_by="prefix", colors=None):
    # Create a color mapping using closure with set for prefix tracking
    if colors is None:
        try:
            cmap = plt.colormaps.get_cmap(colormap)
            # If n is not provided, try to get it from the colormap, or default to 8
            if n is None:
                n = getattr(cmap, "N", 8)
            colors = [cmap(i) for i in linspace(0, 1, n)]
        except ValueError:
            raise ValueError(f"Unknown colormap: {colormap}")

    if cycle_by == "prefix":
        linecolors = {}
        color_iter = iter(colors)

        def get_color(label):
            prefix = get_label_prefix(label)

            # If prefix not seen before, assign new color
            if prefix not in linecolors.keys():
                linecolors[prefix] = next(color_iter)

            return linecolors[prefix]

        return get_color

    if cycle_by == "line":
        linecolors = {}
        color_iter = iter(colors)

        def get_color(label):
            nonlocal color_iter
            if label not in linecolors.keys():
                try:
                    linecolors[label] = next(color_iter)
                except StopIteration:
                    color_iter = iter(colors)
                    return next(color_iter)
            return linecolors[label]

        return get_color


def get_label_from_mapping(metadata_or_str, label_mapping):
    """Extract a clean label from sample ID using regex-based mapping from config with number_range and offset."""
    import re

    # Handle both metadata dict and string input
    if isinstance(metadata_or_str, dict):
        sample_name_full = str(metadata_or_str.get("sample_name", ""))
    else:
        sample_name_full = str(metadata_or_str)

    default_pattern = r"^(?P<prefix>[A-Za-z0-9]*?)[-_]?(?P<number>\d+)$"
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
                            # Handle both single number and range format
                            if "-" in str(range_str):
                                min_num, max_num = map(int, str(range_str).split("-"))
                            else:
                                # Single number case
                                min_num = max_num = int(range_str)
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
                                display_number = offset_idx
                                label = rule["label"].format(number=display_number)
                            return label
    return f"Sample {sample_name_full}"


def match_label_filter(label, filt):
    """Reusable filter function for label matching."""
    if "equals" in filt:
        return (
            label.startswith(filt["equals"])
            and len(label[len(filt["equals"]) :].split(" ")) == 1
        )
    elif "contains" in filt:
        return filt["contains"] in label
    return False


def filter_columns(df, filters):
    """Filter DataFrame columns based on filter criteria"""
    if not filters:
        return df.columns.tolist()

    columns = df.columns.tolist()
    result = []

    for filter_spec in filters:
        if "key" in filter_spec and filter_spec["key"] == "label":
            if "equals" in filter_spec:
                # Use the sophisticated "equals" matching from h5 plot
                # Apply the filter to each column individually
                for col in columns:
                    if match_label_filter(col, filter_spec):
                        result.append(col)
            elif "contains" in filter_spec:
                # Use simple contains matching
                result.extend(
                    [col for col in columns if filter_spec["contains"] in col]
                )

    return list(set(result))  # Remove duplicates


def add_peak_label(
    ax, df, x, text, x_range=1.0, y_offset=None, log_scale=None, fontdict=None
):
    """
    Add a text label to the axis at the given x position.
    The y position is set to the max y value within ±x_range of x across all int_ columns, plus y_offset.

    Parameters:
    - ax: matplotlib axis object
    - df: DataFrame with intensity columns (any column names)
    - x: x position for the label
    - text: text to display
    - x_range: range around x to search for maximum (default 1.0)
    - y_offset: offset for y position (multiplicative for log scale, additive for linear)
    - log_scale: If True, treats y_offset as multiplicative factor. If False, additive offset. If None, auto-detect.
    - fontdict: font properties dictionary
    """
    if log_scale is None:
        log_scale = ax.get_yscale() == "log"

    if y_offset is None:
        y_offset = 1.2 if log_scale else 20

    # Filter rows within the x range
    mask = (df.index > (x - x_range)) & (df.index < (x + x_range))
    if mask.any():
        # Get all intensity columns (all columns in this case)
        int_cols = df.columns.tolist()
        # Find the maximum y value across all intensity columns within the range
        y_max = df.loc[mask, int_cols].max().max()

        if log_scale:
            # For log scale, multiply by offset factor to get consistent visual spacing
            y_text = y_max * y_offset
        else:
            # For linear scale, add offset
            y_text = y_max + y_offset

        ax.text(
            x, y_text, text, clip_on=True, fontdict=fontdict
        )  # Use fontdict if provided
    else:
        fallback_y = y_offset if not log_scale else y_offset
        ax.text(
            x, fallback_y, text, clip_on=True, fontdict=fontdict
        )  # fallback if no data in range


def voigt(x, amplitude, center, sigma, gamma, baseline):
    """Voigt profile function for peak fitting"""
    from scipy.special import wofz
    import numpy as np

    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi)) + baseline


def voigt_fwhm(sigma, gamma):
    """Calculate FWHM for Voigt profile"""
    import numpy as np

    fwhm_gauss = 2.355 * sigma  # 2*sqrt(2*ln(2)) * sigma
    fwhm_lorentz = 2 * gamma
    # Approximation for Voigt FWHM
    fwhm_voigt = 0.5346 * fwhm_lorentz + np.sqrt(
        0.2166 * fwhm_lorentz**2 + fwhm_gauss**2
    )
    return fwhm_voigt


def fit_peak_fwhm(x_data, y_data, peak_center_guess):
    """Calculate FWHM for a single peak using Voigt profile fitting with constraints"""
    from scipy.optimize import curve_fit
    import numpy as np

    try:
        # Constrain fitting to the peak region (-110 to -70 degrees)
        peak_region_mask = (x_data >= -110) & (x_data <= -70)

        if not np.any(peak_region_mask):
            return {"success": False, "error": "No data in peak region (-110 to -70)"}

        x_fit = x_data[peak_region_mask]
        y_fit = y_data[peak_region_mask]

        # Remove baseline by finding minimum in a wider region
        baseline_region_mask = (x_data >= -120) & (x_data <= -60)
        if np.any(baseline_region_mask):
            baseline_guess = np.percentile(
                y_data[baseline_region_mask], 10
            )  # Use 10th percentile as baseline
        else:
            baseline_guess = np.min(y_fit)

        # Subtract baseline for better peak detection
        y_fit_corrected = y_fit - baseline_guess

        # Find actual peak center in the constrained region
        peak_idx = np.argmax(y_fit_corrected)
        center_guess = x_fit[peak_idx]

        # Better initial parameter guesses
        amplitude_guess = np.max(y_fit_corrected)
        sigma_guess = 2.0  # Reasonable initial width for XRD peaks
        gamma_guess = 1.0  # Initial Lorentzian component

        initial_guess = [
            amplitude_guess,
            center_guess,
            sigma_guess,
            gamma_guess,
            baseline_guess,
        ]

        # Set parameter bounds to constrain the fit
        # [amplitude, center, sigma, gamma, baseline]
        lower_bounds = [0, -110, 0.1, 0.1, 0]  # Center must be in peak region
        upper_bounds = [np.inf, -70, 20, 20, np.inf]  # Reasonable physical limits

        # Fit Voigt profile with constraints
        popt, pcov = curve_fit(
            voigt,
            x_fit,
            y_fit,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
        amplitude, center, sigma, gamma, baseline = popt

        # Calculate FWHM
        fwhm = voigt_fwhm(sigma, gamma)

        # Calculate R² and other quality metrics
        y_pred = voigt(x_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Additional quality check: peak should be within expected range
        if not (-110 <= center <= -70):
            return {
                "success": False,
                "error": f"Peak center {center:.2f} outside expected range (-110 to -70)",
            }

        # Check if FWHM is reasonable (not too narrow or too wide)
        if not (0.5 <= fwhm <= 50):
            return {
                "success": False,
                "error": f"FWHM {fwhm:.3f} outside reasonable range (0.5 to 50)",
            }

        return {
            "success": True,
            "center": center,
            "fwhm": fwhm,
            "r_squared": r_squared,
            "fit_params": popt,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "y_pred": y_pred,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

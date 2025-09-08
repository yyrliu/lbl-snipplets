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
        color_iter = iter(colors)

        def get_color(_):
            nonlocal color_iter
            try:
                return next(color_iter)
            except StopIteration:
                color_iter = iter(colors)
                return next(color_iter)

        return get_color

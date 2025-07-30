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


def get_linestyle_factory(ignore_out_of_range=False):
    # Create a linestyle mapping using closure with dict for prefix tracking
    linestyles = ["-", "--", ":", "-."]
    prefix_iters = {}

    def get_linestyle(label):
        prefix = get_label_prefix(label)

        try:
            return next(prefix_iters[prefix])
        except KeyError:
            # If prefix not seen before, create a new iterator for linestyles
            print(f"Creating new linestyle iterator for prefix: {prefix}")
            prefix_iters[prefix] = iter(linestyles)
            return next(prefix_iters[prefix])
        except StopIteration as e:
            if not ignore_out_of_range:
                raise e

            prefix_iters[prefix] = iter(linestyles)
            return next(prefix_iters[prefix])

    return get_linestyle


def get_color_factory(
    colormap="qualitative", n=None, cycle_by_prefix=True, colors=None
):
    # Create a color mapping using closure with set for prefix tracking
    if colors is None:
        if colormap == "qualitative":
            colors = plt.colormaps.get_cmap("tab10").colors
        elif colormap == "sequential":
            colors = [
                plt.colormaps.get_cmap("viridis").reversed()(i)
                for i in linspace(0, 1, n or 6)
            ]
        else:
            raise ValueError(f"Unknown colormap: {colormap}")

    if cycle_by_prefix:
        prefix_colors = {}
        used_prefixes = set()

        def get_color(label):
            prefix = get_label_prefix(label)

            # If prefix not seen before, assign new color
            if prefix not in used_prefixes:
                used_prefixes.add(prefix)
                color_index = len(used_prefixes) - 1
                prefix_colors[prefix] = colors[color_index % len(colors)]

            return prefix_colors[prefix]

        return get_color

    else:
        color_iter = iter(colors)

        def get_color(_):
            return next(color_iter)

        return get_color

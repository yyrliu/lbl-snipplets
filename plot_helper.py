import matplotlib.pyplot as plt

def get_label_prefix(label):
    # Extract prefix from label (everything before the first space or underscore)
    if " " in label:
        return label.split(" ")[0]
    elif "_" in label:
        return label.split("_")[0]
    else:
        return label

def get_linestyle_factory():
    # Create a linestyle mapping using closure with dict for prefix tracking
    linestyles = ["-", "--", ":", "-."]
    used_prefixes = set()
    linestyle_iter = iter(linestyles)

    def get_linestyle(label):
        nonlocal linestyle_iter
        prefix = get_label_prefix(label)

        # If prefix not seen before, assign new linestyle
        if prefix not in used_prefixes:
            used_prefixes.add(prefix)
            linestyle_iter = iter(linestyles)

        return next(linestyle_iter)

    return get_linestyle


def get_color_factory(colormap=None, cycle_by_prefix=True):
    # Create a color mapping using closure with set for prefix tracking
    if colormap is None:
        colormap = plt.colormaps.get_cmap("tab10").colors

    prefix_colors = {}
    used_prefixes = set()

    if cycle_by_prefix:

        def get_color(label):
            prefix = get_label_prefix(label)

            # If prefix not seen before, assign new color
            if prefix not in used_prefixes:
                used_prefixes.add(prefix)
                color_index = len(used_prefixes) - 1
                prefix_colors[prefix] = colormap[color_index % len(colormap)]

            return prefix_colors[prefix]

        return get_color

    else:
        color_iter = iter(colormap)

        def get_color(label):
            nonlocal color_iter
            return next(color_iter)

        return get_color
    
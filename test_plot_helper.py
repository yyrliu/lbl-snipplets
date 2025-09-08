import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_helper import (
    get_label_from_mapping,
    match_label_filter,
    filter_columns,
    add_peak_label,
)


def test_get_label_from_mapping_with_string():
    """Test get_label_from_mapping with string input."""
    label_mapping = [
        {
            "pattern": r"^(?P<prefix>[A-Za-z0-9]*?)[-_]?(?P<number>\d+)$",
            "rules": [
                {"prefix": "", "number_range": "1-2", "label": "0.3M-wash {number}"},
                {"prefix": "", "number_range": "3-4", "label": "0.3M {number}"},
                {"prefix": "", "number_range": "5-6", "label": "1.0M-wash {number}"},
                {"prefix": "", "number_range": "7-8", "label": "1.0M {number}"},
                {
                    "prefix": "",
                    "number_range": "9-12",
                    "label": "Anneal {mapped_result} min (0.3M, no PMMA)",
                    "number_mapping": {1: 20, 2: 17, 3: 13, 4: 10},
                },
            ],
        }
    ]

    # Test simple number mapping
    assert get_label_from_mapping("1", label_mapping) == "0.3M-wash 1"
    assert get_label_from_mapping("2", label_mapping) == "0.3M-wash 2"
    assert get_label_from_mapping("3", label_mapping) == "0.3M 1"
    assert get_label_from_mapping("4", label_mapping) == "0.3M 2"
    assert get_label_from_mapping("5", label_mapping) == "1.0M-wash 1"
    assert get_label_from_mapping("6", label_mapping) == "1.0M-wash 2"
    assert get_label_from_mapping("7", label_mapping) == "1.0M 1"
    assert get_label_from_mapping("8", label_mapping) == "1.0M 2"

    # Test number mapping with custom mapping
    assert get_label_from_mapping("9", label_mapping) == "Anneal 20 min (0.3M, no PMMA)"
    assert (
        get_label_from_mapping("10", label_mapping) == "Anneal 17 min (0.3M, no PMMA)"
    )
    assert (
        get_label_from_mapping("11", label_mapping) == "Anneal 13 min (0.3M, no PMMA)"
    )
    assert (
        get_label_from_mapping("12", label_mapping) == "Anneal 10 min (0.3M, no PMMA)"
    )

    # Test out of range
    assert get_label_from_mapping("99", label_mapping) == "Sample 99"


def test_get_label_from_mapping_with_metadata_dict():
    """Test get_label_from_mapping with metadata dictionary input."""
    label_mapping = [
        {
            "pattern": r"^(?P<prefix>[A-Za-z0-9]*?)[-_]?(?P<number>\d+)$",
            "rules": [
                {"prefix": "", "number_range": "1-2", "label": "0.3M-wash {number}"}
            ],
        }
    ]

    metadata = {"sample_name": "1"}
    assert get_label_from_mapping(metadata, label_mapping) == "0.3M-wash 1"

    # Test missing sample_name
    metadata_empty = {}
    assert get_label_from_mapping(metadata_empty, label_mapping) == "Sample "


def test_get_label_from_mapping_with_prefix():
    """Test get_label_from_mapping with prefix patterns."""
    label_mapping = [
        {
            "pattern": r"^(?P<prefix>[A-Za-z0-9]+)[-_]?(?P<number>\d+)$",
            "rules": [
                {"prefix": "GB", "number_range": "1-4", "label": "GB {number}"},
                {"prefix": "Robot", "number_range": "1-4", "label": "Robot {number}"},
            ],
        }
    ]

    assert get_label_from_mapping("GB1", label_mapping) == "GB 1"
    assert get_label_from_mapping("GB_2", label_mapping) == "GB 2"
    assert get_label_from_mapping("Robot-3", label_mapping) == "Robot 3"
    assert get_label_from_mapping("Robot4", label_mapping) == "Robot 4"

    # Test unmatched prefix
    assert get_label_from_mapping("Unknown1", label_mapping) == "Sample Unknown1"


def test_match_label_filter():
    """Test match_label_filter function for various filter types."""

    # Test equals filter with prefix and digit
    equals_filter = {"key": "label", "equals": "0.3M "}
    assert match_label_filter("0.3M 1", equals_filter) == True
    assert match_label_filter("0.3M 2", equals_filter) == True
    assert match_label_filter("0.3M-wash 1", equals_filter) == False  # Different prefix
    assert match_label_filter("1.0M 1", equals_filter) == False  # Different prefix
    assert match_label_filter("0.3M", equals_filter) == False  # No digit after
    assert match_label_filter("0.3M abc", equals_filter) == False  # Not a digit

    # Test contains filter
    contains_filter = {"key": "label", "contains": "Anneal"}
    assert match_label_filter("Anneal 20 min (0.3M, no PMMA)", contains_filter) == True
    assert match_label_filter("Anneal 17 min (0.3M, no PMMA)", contains_filter) == True
    assert match_label_filter("0.3M 1", contains_filter) == False

    # Test unknown filter type
    unknown_filter = {"key": "label", "unknown": "test"}
    assert match_label_filter("anything", unknown_filter) == False


def test_filter_columns():
    """Test filter_columns function with DataFrame."""
    import pandas as pd

    # Create test DataFrame
    df = pd.DataFrame(
        {
            "0.3M 1": [1, 2, 3],
            "0.3M 2": [4, 5, 6],
            "1.0M 1": [7, 8, 9],
            "1.0M 2": [10, 11, 12],
            "Anneal 20 min (0.3M, no PMMA)": [13, 14, 15],
            "Anneal 17 min (0.3M, no PMMA)": [16, 17, 18],
        }
    )

    # Test filtering for 0.3M samples
    filters_03m = [{"key": "label", "equals": "0.3M "}]
    result = filter_columns(df, filters_03m)
    expected = ["0.3M 1", "0.3M 2"]
    assert set(result) == set(expected)  # Use set comparison to ignore order

    # Test filtering for 1.0M samples
    filters_10m = [{"key": "label", "equals": "1.0M "}]
    result = filter_columns(df, filters_10m)
    expected = ["1.0M 1", "1.0M 2"]
    assert set(result) == set(expected)  # Use set comparison to ignore order

    # Test filtering for Anneal samples
    filters_anneal = [{"key": "label", "contains": "Anneal"}]
    result = filter_columns(df, filters_anneal)
    expected = ["Anneal 20 min (0.3M, no PMMA)", "Anneal 17 min (0.3M, no PMMA)"]
    assert set(result) == set(expected)  # Use set comparison to ignore order

    # Test multiple filters (OR logic)
    filters_multiple = [
        {"key": "label", "equals": "0.3M "},
        {"key": "label", "equals": "1.0M "},
    ]
    result = filter_columns(df, filters_multiple)
    expected = ["0.3M 1", "0.3M 2", "1.0M 1", "1.0M 2"]
    assert set(result) == set(expected)  # Use set comparison to ignore order

    # Test no matches
    filters_none = [{"key": "label", "equals": "nonexistent "}]
    result = filter_columns(df, filters_none)
    assert result == []


def test_add_peak_label():
    """Test peak label positioning function"""
    # Use non-interactive backend for testing
    import matplotlib

    matplotlib.use("Agg")

    # Create test data
    x = np.linspace(0, 10, 100)
    y1 = np.exp(-((x - 3) ** 2) / 0.5) * 100 + 10  # Peak at x=3
    y2 = np.exp(-((x - 7) ** 2) / 0.3) * 200 + 10  # Peak at x=7

    df = pd.DataFrame({"sample1": y1, "sample2": y2}, index=x)

    # Create a test plot
    fig, ax = plt.subplots()
    ax.plot(df.index, df["sample1"], label="sample1")
    ax.plot(df.index, df["sample2"], label="sample2")
    ax.set_yscale("log")

    # Test adding peak label - should not raise errors
    add_peak_label(ax, df, 3.0, "Peak 1", x_range=0.5)
    add_peak_label(ax, df, 7.0, "Peak 2", x_range=0.5, y_offset=1.5)

    # Check that text objects were added
    texts = [child for child in ax.get_children() if hasattr(child, "get_text")]
    text_labels = [t.get_text() for t in texts if hasattr(t, "get_text")]

    # Should have our peak labels plus any legend text
    assert "Peak 1" in text_labels
    assert "Peak 2" in text_labels

    plt.close(fig)


def test_add_peak_label_linear_scale():
    """Test peak label positioning with linear scale"""
    # Use non-interactive backend for testing
    import matplotlib

    matplotlib.use("Agg")

    # Create test data
    x = np.linspace(0, 10, 100)
    y1 = np.exp(-((x - 5) ** 2) / 0.5) * 100 + 10

    df = pd.DataFrame({"sample1": y1}, index=x)

    # Create a test plot with linear scale
    fig, ax = plt.subplots()
    ax.plot(df.index, df["sample1"])
    # Keep linear scale (default)

    # Test adding peak label with linear scale
    add_peak_label(ax, df, 5.0, "Linear Peak", x_range=0.5, y_offset=20)

    # Should not raise errors
    texts = [child for child in ax.get_children() if hasattr(child, "get_text")]
    text_labels = [t.get_text() for t in texts if hasattr(t, "get_text")]
    assert "Linear Peak" in text_labels

    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])

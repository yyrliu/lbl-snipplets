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


def test_get_label_from_mapping_single_number_range():
    """Test get_label_from_mapping with single number number_range."""
    label_mapping = [
        {
            "pattern": r"^(?P<prefix>[A-Za-z0-9]*?)[-_]?(?P<number>\d+)$",
            "rules": [
                # Test single number as string
                {"prefix": "", "number_range": "3", "label": "Single sample {number}"},
                # Test single number as integer
                {"prefix": "", "number_range": 5, "label": "Integer single {number}"},
                # Test range format (for comparison)
                {"prefix": "", "number_range": "7-9", "label": "Range {number}"},
                # Test single number with number_mapping
                {
                    "prefix": "",
                    "number_range": "10",
                    "label": "Mapped single {mapped_result}",
                    "number_mapping": {1: 42},
                },
            ],
        }
    ]

    # Test single number string format
    assert get_label_from_mapping("3", label_mapping) == "Single sample 1"
    assert get_label_from_mapping("2", label_mapping) == "Sample 2"  # Not in range
    assert get_label_from_mapping("4", label_mapping) == "Sample 4"  # Not in range

    # Test single number integer format
    assert get_label_from_mapping("5", label_mapping) == "Integer single 1"
    assert get_label_from_mapping("6", label_mapping) == "Sample 6"  # Not in range

    # Test range format still works
    assert get_label_from_mapping("7", label_mapping) == "Range 1"
    assert get_label_from_mapping("8", label_mapping) == "Range 2"
    assert get_label_from_mapping("9", label_mapping) == "Range 3"

    # Test single number with number_mapping
    assert get_label_from_mapping("10", label_mapping) == "Mapped single 42"

    # Test prefixed single numbers
    label_mapping_prefix = [
        {
            "pattern": r"^(?P<prefix>[A-Za-z0-9]+)[-_]?(?P<number>\d+)$",
            "rules": [
                {"prefix": "GB", "number_range": "2", "label": "GB single {number}"},
                {
                    "prefix": "Robot",
                    "number_range": 4,
                    "label": "Robot single {number}",
                },
            ],
        }
    ]

    assert get_label_from_mapping("GB2", label_mapping_prefix) == "GB single 1"
    assert (
        get_label_from_mapping("GB3", label_mapping_prefix) == "Sample GB3"
    )  # Not in range
    assert get_label_from_mapping("Robot4", label_mapping_prefix) == "Robot single 1"
    assert (
        get_label_from_mapping("Robot5", label_mapping_prefix) == "Sample Robot5"
    )  # Not in range


def test_get_label_from_mapping_comma_separated_range():
    """Test get_label_from_mapping with comma-separated number ranges."""
    label_mapping = [
        {
            "pattern": r"^(?P<prefix>[A-Za-z0-9]*?)[-_]?(?P<number>\d+)$",
            "rules": [
                # Test comma-separated values
                {"prefix": "", "number_range": "1, 3, 5", "label": "0.3M-wash {number}"},
                {"prefix": "", "number_range": "2, 4, 6", "label": "1.0M-wash {number}"},
                {"prefix": "", "number_range": "7, 9, 11", "label": "Anneal {number} min"},
                # Test comma-separated with number_mapping
                {
                    "prefix": "",
                    "number_range": "10, 15, 20",
                    "label": "Custom {mapped_result}",
                    "number_mapping": {1: "A", 2: "B", 3: "C"},
                },
                # Test mixed with spaces and no spaces
                {"prefix": "", "number_range": "12,13, 14", "label": "Mixed spacing {number}"},
            ],
        }
    ]

    # Test comma-separated values - should get position-based numbering
    assert get_label_from_mapping("1", label_mapping) == "0.3M-wash 1"  # First position
    assert get_label_from_mapping("3", label_mapping) == "0.3M-wash 2"  # Second position
    assert get_label_from_mapping("5", label_mapping) == "0.3M-wash 3"  # Third position
    
    assert get_label_from_mapping("2", label_mapping) == "1.0M-wash 1"  # First position
    assert get_label_from_mapping("4", label_mapping) == "1.0M-wash 2"  # Second position
    assert get_label_from_mapping("6", label_mapping) == "1.0M-wash 3"  # Third position

    assert get_label_from_mapping("7", label_mapping) == "Anneal 1 min"   # First position
    assert get_label_from_mapping("9", label_mapping) == "Anneal 2 min"   # Second position
    assert get_label_from_mapping("11", label_mapping) == "Anneal 3 min"  # Third position

    # Test comma-separated with custom number_mapping
    assert get_label_from_mapping("10", label_mapping) == "Custom A"  # maps to position 1 -> "A"
    assert get_label_from_mapping("15", label_mapping) == "Custom B"  # maps to position 2 -> "B"
    assert get_label_from_mapping("20", label_mapping) == "Custom C"  # maps to position 3 -> "C"

    # Test mixed spacing in comma-separated values
    assert get_label_from_mapping("12", label_mapping) == "Mixed spacing 1"
    assert get_label_from_mapping("13", label_mapping) == "Mixed spacing 2"
    assert get_label_from_mapping("14", label_mapping) == "Mixed spacing 3"

    # Test numbers not in comma-separated list
    assert get_label_from_mapping("8", label_mapping) == "Sample 8"   # Not in any list
    assert get_label_from_mapping("16", label_mapping) == "Sample 16" # Not in any list
    assert get_label_from_mapping("99", label_mapping) == "Sample 99" # Not in any list


def test_get_label_from_mapping_comma_separated_with_prefix():
    """Test get_label_from_mapping with comma-separated ranges and prefixes."""
    label_mapping = [
        {
            "pattern": r"^(?P<prefix>[A-Za-z0-9]+)[-_](?P<number>\d+)$",  # Require separator for clear parsing
            "rules": [
                {"prefix": "GB", "number_range": "1, 3, 5", "label": "GB Sample {number}"},
                {"prefix": "Robot", "number_range": "2, 4, 6", "label": "Robot {number}"},
                {
                    "prefix": "Test",
                    "number_range": "10, 20, 30",
                    "label": "Test {mapped_result}",
                    "number_mapping": {1: "Alpha", 2: "Beta", 3: "Gamma"},
                },
            ],
        }
    ]

    # Test prefixed comma-separated values (using separators for clear parsing)
    assert get_label_from_mapping("GB-1", label_mapping) == "GB Sample 1"
    assert get_label_from_mapping("GB_3", label_mapping) == "GB Sample 2"
    assert get_label_from_mapping("GB-5", label_mapping) == "GB Sample 3"
    assert get_label_from_mapping("GB-2", label_mapping) == "Sample GB-2"  # Not in list

    assert get_label_from_mapping("Robot_2", label_mapping) == "Robot 1"
    assert get_label_from_mapping("Robot-4", label_mapping) == "Robot 2"
    assert get_label_from_mapping("Robot_6", label_mapping) == "Robot 3"
    assert get_label_from_mapping("Robot-1", label_mapping) == "Sample Robot-1"  # Not in list

    # Test with custom mapping
    assert get_label_from_mapping("Test-10", label_mapping) == "Test Alpha"
    assert get_label_from_mapping("Test_20", label_mapping) == "Test Beta"
    assert get_label_from_mapping("Test-30", label_mapping) == "Test Gamma"
    assert get_label_from_mapping("Test_15", label_mapping) == "Sample Test_15"  # Not in list


def test_get_label_from_mapping_all_formats_combined():
    """Test get_label_from_mapping with all supported formats in one mapping."""
    label_mapping = [
        {
            "pattern": r"^(?P<prefix>[A-Za-z0-9]*?)[-_]?(?P<number>\d+)$",
            "rules": [
                # Single number
                {"prefix": "", "number_range": "1", "label": "Single {number}"},
                # Range format
                {"prefix": "", "number_range": "2-4", "label": "Range {number}"},
                # Comma-separated
                {"prefix": "", "number_range": "5, 7, 9", "label": "Comma {number}"},
                # Mixed with number_mapping
                {
                    "prefix": "",
                    "number_range": "10, 12, 14",
                    "label": "Mapped {mapped_result}",
                    "number_mapping": {1: 100, 2: 200, 3: 300},
                },
            ],
        }
    ]

    # Test single number format
    assert get_label_from_mapping("1", label_mapping) == "Single 1"

    # Test range format
    assert get_label_from_mapping("2", label_mapping) == "Range 1"
    assert get_label_from_mapping("3", label_mapping) == "Range 2"
    assert get_label_from_mapping("4", label_mapping) == "Range 3"

    # Test comma-separated format
    assert get_label_from_mapping("5", label_mapping) == "Comma 1"
    assert get_label_from_mapping("7", label_mapping) == "Comma 2"
    assert get_label_from_mapping("9", label_mapping) == "Comma 3"

    # Test comma-separated with mapping
    assert get_label_from_mapping("10", label_mapping) == "Mapped 100"
    assert get_label_from_mapping("12", label_mapping) == "Mapped 200"
    assert get_label_from_mapping("14", label_mapping) == "Mapped 300"

    # Test numbers not matching any format
    assert get_label_from_mapping("6", label_mapping) == "Sample 6"
    assert get_label_from_mapping("8", label_mapping) == "Sample 8"
    assert get_label_from_mapping("11", label_mapping) == "Sample 11"


def test_match_label_filter():
    """Test match_label_filter function for various filter types."""

    # Test equals filter with prefix and digit or tag
    equals_filter = {"key": "label", "equals": "0.3M "}
    assert match_label_filter("0.3M 1", equals_filter)
    assert match_label_filter("0.3M 2", equals_filter)
    assert match_label_filter("0.3M abc", equals_filter)
    assert not match_label_filter("0.3M-wash 1", equals_filter)  # Different prefix
    assert not match_label_filter("1.0M 1", equals_filter)  # Different prefix
    assert not match_label_filter("0.3M", equals_filter)  # No digit after

    # Test contains filter
    contains_filter = {"key": "label", "contains": "Anneal"}
    assert match_label_filter("Anneal 20 min (0.3M, no PMMA)", contains_filter)
    assert match_label_filter("Anneal 17 min (0.3M, no PMMA)", contains_filter)
    assert not match_label_filter("0.3M 1", contains_filter)

    # Test unknown filter type
    unknown_filter = {"key": "label", "unknown": "test"}
    assert not match_label_filter("anything", unknown_filter)


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

import h5py
import numpy as np
import cv2
from scipy.stats import entropy

# source: https://github.com/malghal/perovskites_antisolvents_clustering/blob/467a7789e5b0e529fc69dfb31acb9ad4492cd809/Codes/BO/generating_total_score_components_unused.ipynb

# This function take in raw image from the h5 file extrated and cut the plate holder.  Accept numpy array directly instead of file path
# Input: Raw image (numpy.array)
# Return: Cut image (numpy.array)
def batch_processing(raw_image_array):
    """
    Input: Raw image (numpy.array) - directly from h5 file
    Return: Cut image (numpy.array)
    """
    # If it's already grayscale, use as-is; if RGB, convert to grayscale
    if len(raw_image_array.shape) == 3:
        gray_image = cv2.cvtColor(raw_image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = raw_image_array

    # Apply the same cropping
    image_cut_full = gray_image[500:2400, 1300:3100]
    return image_cut_full


# This function take in the image and using standardize and normalize method to 0 and 1
# Input: Image (numpy.array)
# Output: Normalized image (numpy.array)
def standardize(image):
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    # Calculate for standardized image
    standardized_img = (image - mean) / std
    min_val = np.min(standardized_img)
    max_val = np.max(standardized_img)
    # Calculate for normalized image
    normalized_img = (standardized_img - min_val) / (max_val - min_val)
    return normalized_img


# This function is take in image and calculate dispersion of the image
# Input: Image (numpy.array)
# Output: Standard deviation score
def STD_PL(image):
    pixels = np.array(image)
    std_dev_normalized = np.std(pixels)
    return std_dev_normalized


# This function is take in image and calculate the randomness of the image
# Input: Image (numpy.array)
# Output: Entropy score
def entropy_value_1(image):
    pixels = np.array(image)
    total_pixels = np.size(image)
    # Convert it into histogram from 0 - 1
    hist, _ = np.histogram(pixels, bins=101, range=(0, 1), density=True)
    hist_normalized = hist / total_pixels
    # Caculate entropy base on bits
    image_entropy = entropy(hist_normalized, base=2)
    return image_entropy


# This function get all the ring starting from the center to the boundary
# Input: Image (numpy.array)
# Output: intensity radius list (numpy.array)
def radius_intensity(image):
    # Get center point of the image
    width, height = image.shape
    x = round(width / 2)
    y = round(height / 2)
    center_coordinates = (x, y)

    # Define radius and ring
    radius = 50
    ring = 1
    intensity_rad_list = []
    radius_list = []

    # first circle
    radius_list.append(f"{ring}")
    mask_1 = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask_1, center_coordinates, radius, 255, -1)
    pixel_255_locations = np.column_stack(np.where(mask_1 == 255))
    inside_circle = image[pixel_255_locations[:, 0], pixel_255_locations[:, 1]]
    average_value = np.average(inside_circle)
    intensity_rad_list.append(average_value)
    radius += 50
    ring += 1

    # recursive until reach the boundary for detect circle
    while radius <= x and radius <= y:
        radius_list.append(f"{ring}")
        mask_2 = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask_2, center_coordinates, radius, 255, -1)
        ring_between = cv2.subtract(mask_2, mask_1)
        pixel_255_locations = np.column_stack(np.where(ring_between == 255))
        inside_circle = image[pixel_255_locations[:, 0], pixel_255_locations[:, 1]]
        average_value = np.average(inside_circle)
        intensity_rad_list.append(average_value)
        radius += 50
        ring += 1
        mask_1 = mask_2
    return np.array(intensity_rad_list)


# This function find the location of all extrema in intenisty radius list
# Input: List of intenisty score
# Output: Location of extrema occur.
def find_local_extrema_with_threshold(data, threshold=0):
    # Compute the first derivative's sign change
    gradient = np.diff(np.sign(np.diff(data)))
    # Maxima: Where the gradient changes from positive to negative (< 0)
    maxima_indices = np.where(gradient < 0)[0] + 1
    # Minima: Where the gradient changes from negative to positive (> 0)
    minima_indices = np.where(gradient > 0)[0] + 1
    # Combine and sort extrema
    all_extrema = np.sort(
        np.concatenate([[0, np.size(data) - 1], maxima_indices, minima_indices])
    )
    # Filter extrema based on the vertical threshold
    filtered_extrema = [all_extrema[0]]  # Start with the first extremum
    # recursive over list and filter out any extrema less than the threshold
    for i in range(1, len(all_extrema) - 1):
        prev = filtered_extrema[-1]
        curr = all_extrema[i]
        next = all_extrema[i + 1]
        # Check if the vertical distance is above the threshold
        if (
            abs(data[curr] - data[prev]) >= threshold
            and abs(data[curr] - data[next]) >= threshold
        ):
            filtered_extrema.append(curr)
    return filtered_extrema + [all_extrema[-1]]


# This function caculate for variance between each extrema
# Input: Extrema list and extrema location
# Return Variance score
def extrema_variance(extrema_list, extrema_location):
    list_variance_score = 0
    number_of_rings = len(extrema_location) - 1
    # recursive over extrema location
    for i in range(0, len(extrema_location) - 1):
        list_extrema = []
        # recursive over extreme location
        for j in range(extrema_location[i], extrema_location[i + 1] + 1):
            # Get all the score of extrema
            list_extrema.append(extrema_list[j])
        # Convert it into numpy to calculate the score
        list_extrema = np.array(list_extrema)
        # Calculate variance using Std functuon
        extrema_variance_score = np.std(list_extrema)
        # Square the variance score and add it to the total variance score
        list_variance_score += extrema_variance_score * extrema_variance_score
    # convert it into numpy
    list_variance_score = np.array(list_variance_score)
    # Square rooot the total score
    total_score = np.sqrt(list_variance_score)
    # Make the score larger by number of rings in the images
    total_score = total_score * number_of_rings
    return total_score


# This function calculate heterogeneity score by normalized and weighted out for 3 features score
# Input: Entropy score, standard deviation score and extrema score
# Output: Heterogeneity score
def weight_avg(entropy_val, STD_value, extrema_score):
    normalize_entropy = entropy_val * 1 / 6.372859213833238
    normalize_STD = STD_value * 1 / 0.295093297958374
    normalize_extrema = extrema_score * 1 / 0.25685287332947604
    final_score = (
        0.1 * normalize_entropy + 0.2 * normalize_STD + 0.7 * normalize_extrema
    )
    return final_score


# This function is call all the functions above it to calcualte for heterogeneity score. Work directly with numpy array
# Input: Unprocess image
# Output: Heterogeneity score
def optimize_PL_image(raw_img_array):
    """
    Input: Raw image numpy array (directly from h5)
    Output: Heterogeneity score
    """
    cutted_img = batch_processing(raw_img_array)
    standardize_img = standardize(cutted_img)
    entro_val = entropy_value_1(standardize_img)
    STD_value = STD_PL(standardize_img)
    array_of_rings = radius_intensity(standardize_img)
    location_of_extrema = find_local_extrema_with_threshold(array_of_rings, 0.02)
    extrema_score = extrema_variance(array_of_rings, location_of_extrema)
    final_score = weight_avg(entro_val, STD_value, extrema_score)
    return final_score


# Modified: No temporary file needed
def my_heterogeneity_score(current_f5_file_path):
    """
    Input: Path to h5 file
    Output: Heterogeneity score (no temporary files)
    """
    with h5py.File(current_f5_file_path, "r") as hf:
        # Check for 'adj_photo' or 'photo'
        if "measurement/spec_run/adj_photo" in hf:
            photo = hf["measurement/spec_run/adj_photo"][:]
            photo_exposure = (
                hf["measurement/spec_run/adj_photo_exposure"][()]
                if "measurement/spec_run/adj_photo_exposure" in hf
                else None
            )
            print("Adjusted photo found")
        elif "measurement/spec_run/photo" in hf:
            photo = hf["measurement/spec_run/photo"][:]
            photo_exposure = (
                hf["measurement/spec_run/photo_exposure"][()]
                if "measurement/spec_run/photo_exposure" in hf
                else None
            )
            print("Regular photo found")
        else:
            raise KeyError("Neither 'adj_photo' nor 'photo' found in the HDF5 file")

        # Ensure proper data type for image processing
        if photo.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            photo_normalized = (
                (photo - photo.min()) / (photo.max() - photo.min()) * 255
            ).astype(np.uint8)
        else:
            photo_normalized = photo

        # Process directly without saving/loading
        heterogeneity_score = optimize_PL_image(photo_normalized)

    return heterogeneity_score

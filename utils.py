import cv2
import os
import numpy as np
import pandas as pd

from osgeo import gdal
from os.path import join
from tqdm import tqdm


def BinarizeRaster(input_raster, focus="dark"):
    """
    Binarizes a raster image based on the mean value, focusing on either dark or light areas.

    Parameters:
    - input_raster: numpy.ndarray, input raster image to binarize.
    - focus: str, focus on "dark" or "light" areas for binarization.

    Returns:
    - numpy.ndarray: Binarized raster image.
    """
    valid_pixels = input_raster[input_raster != -1]
    if valid_pixels.size > 0:
        raster_mean = valid_pixels.mean()
    else:
        raster_mean = 0

    if focus == "light":
        reclass = np.where(input_raster > raster_mean, 1, 0)
    else:
        reclass = np.where((input_raster <= raster_mean)
                           & (input_raster != -1), 1, 0)

    return reclass.astype(np.uint8)


def head_tail_breaks(array, break_per=0.4):
    """
    Applies the head/tail breaks 2.0 to calculate ht-index.

    Parameters:
    - array: numpy.ndarray, the input array for classification.
    - break_per: float, the break percentage to classify the 'head' of the distribution.

    Returns:
    - int: The index at which head/tail breaks classification stops.
    """
    ratio_list = []
    array = array[array > 0]  # Consider only positive values
    ht_index = 1

    while array.size > 1 and np.mean(array) > 0:
        mean_val = np.mean(array)
        head = array[array > mean_val]
        ratio = len(head) / len(array)
        ratio_list.append(ratio)
        if np.mean(ratio_list) > break_per:
            break
        array = head
        ht_index += 1

    return ht_index


def clip(img, lb_images, stats, centers, xy_last, thre=0.00001):
    """
    Clips regions from an image based on the labeled regions and their statistics.

    Parameters:
    - img: numpy.ndarray, the input image to clip from.
    - lb_images: numpy.ndarray, labeled image regions.
    - stats: numpy.ndarray, statistics of each labeled region.
    - centers: numpy.ndarray, centroid of each labeled region.
    - xy_last: numpy.ndarray, last x and y coordinates.
    - thre: float, threshold for selecting large regions.

    Returns:
    - tuple: A tuple containing lists of clipped regions, their centers, and original positions.
    """
    areas = stats[:, cv2.CC_STAT_AREA]
    large_regions_indices = np.where(areas >= thre)[0]

    regions_list, centers_list, original_xy = [], [], []

    for i in large_regions_indices:
        x, y, w, h, _ = stats[i]
        region = img[y:y + h, x:x + w]
        region_lb = lb_images[y:y + h, x:x + w]
        clipped_region = np.where(region_lb == i + 1, region, -1)

        regions_list.append(clipped_region)
        centers_list.append(centers[i].astype(int) + xy_last)
        original_xy.append(np.array([x, y]) + xy_last)

    return regions_list, centers_list, original_xy


def process_hierarchy(regions_list, xy_last_list, focus, break_per=0.4):
    """
    Extract natural cities for one time.

    Parameters:
    - regions_list: list of numpy.ndarray, list of regions/rasters to process.
    - xy_last_list: list of numpy.ndarray, list of last x and y coordinates for each region.
    - focus: str, focus area for binarization ("dark" or "light").
    - break_per: float, threshold percentage for head/tail breaks method.

    Returns:
    - A tuple containing processed region list, diagnostics, and node information.
    """
    return_list, centers_list, xy_list, areas_list = [
    ], [], [], []
    s, d, lr = 0, 0, 0

    with tqdm(total=len(regions_list), desc="Processing regions") as pbar:
        for i, region in enumerate(regions_list):
            bin_image = BinarizeRaster(region, focus)
            num_labels, lb_images, stats, centers = cv2.connectedComponentsWithStats(
                bin_image, connectivity=4)
            num_labels -= 1  # Adjust for background
            stats = stats[1:]  # Ignore background stats
            centers = centers[1:]  # Ignore background center

            # Head/tail breaks on scale rather than area
            areas = np.sqrt(stats[:, cv2.CC_STAT_AREA])
            ht_index = head_tail_breaks(areas, break_per)

            if ht_index > 2:
                regions_list_new, centers_new, xy_new = clip(
                    region, lb_images, stats, centers, xy_last_list[i], thre=0.00001)
                return_list.extend(regions_list_new)
                centers_list.extend(centers_new)
                xy_list.extend(xy_new)
                d += 1
                s += num_labels
                lr += ht_index * num_labels
                areas_list.extend(areas)
            pbar.update(1)

    return return_list, d, s, lr, xy_list


def process_recursively(inputraster, focus):
    """
    Processes an input raster image recursively, generating natural cities at each iteration.

    Parameters:
    - inputraster: numpy.ndarray, the input raster image to process.
    - focus: str, the focus area for binarization ("dark" or "light").

    Returns:
    - list of dicts: A list containing the results from each hierarchical level.
    """
    i = 0
    xy = np.zeros(2, dtype=int)
    results = []

    while True:
        return_list, d, s, lr, xy = process_hierarchy(
            [inputraster] if i == 0 else return_list, xy, focus, break_per=0.4)

        if not return_list:
            break

        print(f"Hierarchy {i+1} has done. D={d}, S={s}, LR={lr}")
        i += 1
        results.append({'output': return_list, 's': s, 'd': d, 'lr': lr,
                        'xy': xy})

    return results


def save_result(csv_path, d_array, s_array, lr_array, i):
    """
    Saves the hierarchical processing results to a CSV file.

    Parameters:
    - csv_path: str, path to the output CSV file.
    - d_array: numpy.ndarray, array of D values for each hierarchy.
    - s_array: numpy.ndarray, array of S values for each hierarchy.
    - lr_array: numpy.ndarray, array of LR values for each hierarchy.
    - i: int, number of hierarchy levels processed.
    """
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))

    h_array = lr_array / s_array
    i_array = np.arange(1, i+1)
    u_array = np.zeros_like(s_array)
    u_array[:-1] = s_array[:-1] - d_array[1:]
    u_array[-1] = s_array[-1]
    per_array = (s_array - u_array) / s_array

    df = pd.DataFrame({
        "I": i_array,
        "D": d_array,
        "S": s_array,
        "H": np.round(h_array, 2),
        "U": u_array.astype(int),
        "%": np.round(per_array, 2),
        "LR": lr_array
    })

    df.to_csv(csv_path, index=False)


def save_subs(output_hie_path, output, inputraster, xy_list, projection, geotransform):
    """
    Saves sub-images from the hierarchical processing to separate files based on their value.

    """

    base = np.zeros_like(inputraster, dtype=np.uint16)

    for i, lr in enumerate(output):
        himg = np.zeros_like(inputraster, dtype=np.uint16)

        for j, current_image in enumerate(lr):
            y, x = xy_list[i][j]
            current_image = np.where(
                current_image > 0, i + 1, 0).astype(np.int16)
            x_end, y_end = x + \
                current_image.shape[0], y + current_image.shape[1]
            himg[x:x_end, y:y_end] = current_image

        base = np.where(himg > 0, himg, base)

    for i in range(1, len(output)+1):
        value_output_path = join(output_hie_path, f"Iteration_{i}.tif")

        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(
            value_output_path, base.shape[1], base.shape[0], 1, gdal.GDT_UInt16)
        output_dataset.SetGeoTransform(geotransform)
        output_dataset.SetProjection(projection)
        output_dataset.GetRasterBand(1).WriteArray(
            np.where(base >= i, 1, 0))
        output_dataset.GetRasterBand(1).SetNoDataValue(0)
        output_dataset.FlushCache()

import os
import numpy as np

from os.path import join
from osgeo import gdal
from utils import process_recursively, save_result, save_subs


def main(data_name, focus='light'):
    """
    Main function to process an image hierarchically and save results.

    Parameters:
    - data_name: str, the input name of your image.
    - focus: str, "light" or "dark" to set the focus for processing.
    """

    # Construct paths in an OS-agnostic way
    image_path = join("Data", f"{data_name}.tif")
    output_csv_path = join("Results_complexity", f"{data_name}.csv")
    output_hie_path = join("Natural_cities", f"{data_name}")

    print("--------------------- Processing ---------------------")

    # Read and preprocess the image
    header = gdal.Open(image_path)
    image = header.ReadAsArray()
    projection = header.GetProjection()
    geotransform = header.GetGeoTransform()
    nodata = header.GetRasterBand(1).GetNoDataValue()
    inputraster = np.where(image == nodata, -1, image).astype(np.int64)

    # Process the image recursively
    results = process_recursively(inputraster, focus)

    # Extract and prepare data for saving results
    i = len(results)
    d_array, s_array, lr_array = (
        np.array([item[key] for item in results]) for key in ('d', 's', 'lr'))
    output_list, xy_list = ([item[key] for item in results]
                            for key in ('output', 'xy'))

    # Calculate and print final metrics
    decs, v, lr = d_array.sum(), i * d_array.sum(), lr_array.sum()
    print("--------------------- Finished ---------------------")
    print(f"Final result: lr = {lr}, V = {v}")

    if not os.path.exists(os.path.dirname(output_csv_path)):
        os.makedirs(os.path.dirname(output_csv_path))

    if not os.path.exists(output_hie_path):
        os.makedirs(output_hie_path)

    # Save results and natural cities
    save_result(output_csv_path, d_array, s_array, lr_array, i)
    save_subs(output_hie_path, output_list, inputraster, xy_list,
              projection, geotransform)


if __name__ == "__main__":

    data_name = 'GD_2022'
    main(data_name)

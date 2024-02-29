# NaturalCitiesfromGlobalNTL

This tutorial delivers a step-by-step tutorial on identifying natural cities using Nighttime Light (NTL) data, those are the brightest areas indicative of high human activity levels. By using the recursive approach, we can delineate not only the boundaries of natural cities but also uncover their inner hotspots of human activity.

## Requirements

The recursive approach is programmed in Python 3. By utilizing the numpy and gdal packages, this implementation significantly outperforms the original ArcPy-based version in speed and memory efficiency, which enables users to delineate natural cities at global scale. You can install all the required environment through:
`pip install -r requirements.txt`

## Data preparation

This tutorial provide a sample NTL imagery (Li et al. 2020) of Guangdong province in China for debugging the program in a short time. Your customized data should be put under the `data` fordler as the sample data.

## Generate natural cities

1. Open python script `generate_nc.py` and adjust the parameters including `data_name` on line 57 based on the sampling year of the NTL imagery used, such as 2000, 2001 etc.
2. Click run button on your IDE or run command:
`python generate_nc.py`
 The program will be automatically generated the natural cities recursively which are save as geotiff format, along with the calculated degree of complexity saved as csv format.
3. Open the output files under the folder nc_tif and results_csv respectively to check whether they are successfully generated. There are I images, where I is the total number of iterations and natural cities or the sub-settlements generated at each iteration is saved with respect to `%data_name%_1.tif`, `%data_name%_2.tif`, …`%data_name%_N.tif`. The complexity results of the NTL imagery will be named as `%data_name%`.csv under the `result` folder.

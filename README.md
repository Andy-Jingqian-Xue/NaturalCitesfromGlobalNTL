# NaturalCitiesfromGlobalNTL

This tutorial delivers a step-by-step tutorial on identifying natural cities using Nighttime Light (NTL) data, those are the brightest areas indicative of high human activity levels. By using the recursive approach, we can delineate not only the boundaries of natural cities but also uncover their inner hotspots of human activity.

## Requirements

The recursive approach is programmed in Python 3. By utilizing the numpy and gdal packages, this implementation significantly outperforms the original ArcPy-based version in speed and memory efficiency, which enables users to delineate natural cities at global scale. You can install all the required environment through:

``` bash
pip install -r requirements.txt
```

## Data preparation

This tutorial provide a sample NTL imagery (Li et al. 2020) of Guangdong province in China for debugging the program in a short time. Your customized data should be put under the `data` fordler as the sample data. If you intend to use `visualize_dynamics.py` to visualize the temporal dynamics of the degree of urban complexity, please ensure your NTL images are named as `2000.tif`,`2001.tif` ..., `2022.tif`.

## Generate natural cities

1. Open python script `generate_nc.py` and adjust the parameters including `data_name` on line 57 based on the sampling year of the NTL imagery used, such as 2000, 2001 etc.
2. Click run button on your IDE or run command:

    ``` bash
    python generate_nc.py
    ```

   The program will be automatically generated the natural cities recursively which are save as geotiff format, along with the calculated degree of complexity saved as csv format.

3. Open the output files under the folder nc_tif and results_csv respectively to check whether they are successfully generated. There are I images, where I is the total number of iterations and natural cities or the sub-settlements generated at each iteration is saved with respect to `%data_name%_1.tif`, `%data_name%_2.tif`, …`%data_name%_N.tif`. The complexity results of the NTL imagery will be named as `%data_name%`.csv under the `result` folder.

## Intepret temporal dynamics of complexity

1. Reconduct generating the natural cities from NTL imagery for the entire 20 years following the steps in last section. You can comment out line 53 in `generate_NC.py`, to only output the statistics of calculating complexity and save memory.
2. Open python script `visualize_dynamics.py`. Click run button on your IDE or run command:

    ``` bash
    python visualize_dynamics.py
    ```

## Results on global NTL imagery in 2022

* Zoomed-in illustration of recursively generating natural cities around the world from the first iteration (a) to the last iteration (f)
![iteration process](https://github.com/AndyXue957/NaturalCitesfromGlobalNTL/blob/main/example_results/iterative%20process.png)

## References

[1] Chen Z., Yu B., Yang C., Zhou Y., Yao S., Qian X., Wang C., Wu B. and Wu J. (2021), An extended time series (2000–2018) of global NPP-VIIRS-Like nighttime light data from a cross-Sensor calibration, _Earth Syst Sci Data_, 13(3), 889-906.

[2] Jiang B. and de Rijke C. (2023), Living images: A recursive approach to computing the structural beauty of images or the livingness of space, _Annals of the American Association of Geographers_, 1-19.

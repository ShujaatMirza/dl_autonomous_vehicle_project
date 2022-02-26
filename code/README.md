# Code
This directory contains code for the final project.

## Dependencies
- `pytorch`
- `torchvision`
- `pyyaml`
- `shapely`
- `pandas`
- `matplotlib`

### Install Dependeicies with conda
- `conda install pyyaml shapely pandas matplotlib pytorch torchvision cudatoolkit=10.0 -c pytorch`

### How to run
- Training
    - Road Map Layout
        ```
        python main.py -config config/road_map.config -mode train
        ```
    - Bounding Box Detection
        ```
        python main.py -config config/bounding_box.config -mode train
        ```
- Testing on our Development Set
    - Road Map Layout
        ```
        python main.py -config config/road_map.config -mode test -checkpoint_path model/roadMapNN_model_at_epoch_10.pt
        ```
    - Bounding Box Detection
        ```
         python main.py -config config/bounding_box.config -mode test -checkpoint_path model/4angle_boundingBoxNN_model_at_epoch_10.pt
        ```
- Running Provided Test Script
    ```
    python run_test.py --data_dir ../data
    ```
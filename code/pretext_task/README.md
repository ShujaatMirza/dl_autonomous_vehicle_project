## Running the pretext task code 

Generate top 500 permutations for jisaw pretext task, found in `permutations_hamming_max_500.npy.` This number can be modified by running the following script with desired value K. 
```
python jigsaw_select_permutations --classes K
```

Train as the intital pretext task as follows:

```
python jigsaw_train.py --config pretext.config
```

For training with integrated model of road_map_laayout, we tried to run using following configurations. 

```
python main.py -config ../config/road_map.config -mode train
```
# Benchmark-on-ForkingPath

To build the testing set, we follow the pipeline in Multiverse to  extract trajectory samples in **ForkingPath** dataset and truncate all future annotations with a maximum 12 steps (2.5 Hz). We only select scenes with bird-eye views and convert all trajectories to real-world coordinates by roughly estimating the ratio of pixel and real-world (4m) length of the vehicles. For the training and validation set, we use the ForkingPath-Anchor in Multiverse and SimAug, a version fully reconstructing VIRAT to CARLA environments with single annotations. Similarly, we use the official splits in SimAug and filter all non-bird-eye-view scenes. Finally, we obtain 11495, 2018, 127 trajectories for training, validation and testing respectively. Following the general settings in pedestrian trajectory prediction, we set the observation and prediction length as 8 (3.2 seconds) and 12 (4.8 seconds) and predict 20 trajectories for each sample. Finally, we carefully select one or two typical methods, where YNet and Multiverse are two models considering scene interactions while others consider social interactions only. We train these models using their release code with their recommended configurations.

<img width="1299" alt="image" src="https://github.com/HRHLALALA/Benchmark-on-ForkingPath/assets/32263355/875a56c1-4c72-4b89-912c-9af8daac98b8">

**Update**
* We released the preprocessed PECNet-format pickles.

**Notebooks will be released soon.**


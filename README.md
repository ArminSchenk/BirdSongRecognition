# Bird Song Recognition
This repository contains the code to replicate the case study from my master thesis "Extending the 'cito' package: deep convolutional neural networks in ecology".

# Data
The data was taken from [BirdCLEF 2024](https://www.kaggle.com/competitions/birdclef-2024/data). Download the "birdclef-2024.zip" file and extract its files into the data/birdclef2024 directory prior to running the analysis. The folder structure should look like this:
```
├──data
|    ├── birdclef2024
|        ├── test_soundscapes
|            └── ...
|        ├── train_audio
|            └── ...
|        ├── unlabeled_soundscapes
|            └── ...
|        ├── eBird_Taxonomy_v2021.csv
|        ├── sample_submission.csv
|        └── train_metadata.csv
```
# Analysis
Run the scripts in the designated order. 
```
├──analysis
|    ├── 1-dataPreparation.R
|    ├── 2-buildingDNNs.R
|    ├── 2-buildingCNNs.R
|    ├── 2-buildingMMNs.R
|    └── 3-visualisation.R
```

# Dinomaly

This is a PyTorch implementation of "Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection". 

In this version, we have made few changes to the model architecture and replaced the original loss function with CosineFocal Loss and added more augmentation for color jitter, brightness, gaussian noice and add. 

Please note that the code is fork of its preview version of original [DinomalyRepo](https://github.com/guojiajeremy/Dinomaly)

## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n my_env python=3.8.12
conda activate my_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA GeForce RTX 3070 (8GB). Same GPU and package version are recommended. 

## 2. Prepare Datasets
Noted that `../` is the upper directory of Dinomaly code. It is where we keep all the datasets by default.
You can also alter it according to your need, just remember to modify the `data_path` in the code. 

### MVTec AD

Download the MVTec-AD dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Unzip the file to `../mvtec_anomaly_detection`.
```
|-- mvtec_anomaly_detection
    |-- bottle
    |-- cable
    |-- capsule
    |-- ....
```


## 3. Run Experiments
Multi-Class Setting
```
python dinomaly_mvtec_uni.py --data_path ../mvtec_anomaly_detection
```


Conventional Class-Separted Setting
```
python dinomaly_mvtec_sep.py --data_path ../mvtec_anomaly_detection
```


Training Unstability: The optimization can be unstable with loss spikes (e.g. ...0.05, 0.04, 0.04, **0.32**, **0.23**, 0.08...)
, which can be harmful to performance. This occurs very very rare. If you see such loss spikes during training, consider change a random seed.

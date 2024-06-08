# Robust-Dinomaly 

the code is fork of its preview version of original [DinomalyRepo](https://github.com/guojiajeremy/Dinomaly). 
 <br>
In this version, we have made few changes to the model architecture and number of layers and replaced the original loss function with CosineFocal Loss and added more augmentation for color jitter, brightness, gaussian noice, random flip and random rotation. I will update the documentation soon with appropriate images.
<br>
<br>
Note: 
1. My Implementation of this paper using  [Ader](https://github.com/zhangzjn/ader) resulted in 62.xx (-2) less Pixel F1 score So I used their official repo for VAND2.0 MvTec AD submission on the last day of VAND2.0.

# Results

Here is the information presented as a table in Markdown for your GitHub README, along with the calculated averages:

---

**MvTec Dataset F1 Scores on Strong Augmentations**

| Category     | Image F1 Max Score | Pixel F1 Max Score |
|--------------|---------------------|--------------------|
| Carpet       | 0.9560              | 0.6144             |
| Grid         | 0.9636              | 0.3670             |
| Leather      | 1.0000              | 0.5062             |
| Tile         | 1.0000              | 0.7447             |
| Wood         | 0.9831              | 0.5501             |
| Bottle       | 1.0000              | 0.7626             |
| Cable        | 0.9405              | 0.5938             |
| Capsule      | 0.9279              | 0.4322             |
| Hazelnut     | 1.0000              | 0.7344             |
| Metal Nut    | 1.0000              | 0.7838             |
| Pill         | 0.9648              | 0.7577             |
| Screw        | 0.8551              | 0.3044             |
| Toothbrush   | 0.9677              | 0.5796             |
| Transistor   | 0.8706              | 0.4397             |
| Zipper       | 0.9367              | 0.3692             |
| Avg          | 0.9577              | 0.5693            |

**Average Scores**

- Average Image F1 Max Score: 0.9577
- Average Pixel F1 Max Score: 0.5693

---

This format should be suitable for inclusion in your GitHub README.


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

## 4. Evaluation for VAND2.0
```
python evaluation.py --module_path ensemble --class_name EnsembleModel --weights_path weights/ --dataset_path ../../datasets/MVTec --category bottle
```
<br>
You can also run all the categories by using

```
bash eval.sh
```
Change the directory to vand2.0_submissions and run thses commands ny setting correct dataset path 

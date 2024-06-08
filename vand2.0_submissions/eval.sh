#!/bin/bash

# Define an array of class names
classes=('carpet' 'grid' 'leather' 'tile' 'wood'
         'bottle' 'cable' 'capsule' 'hazelnut' 'metal_nut'
         'pill' 'screw' 'toothbrush' 'transistor' 'zipper')

# Loop through the array
for class_name in "${classes[@]}"
do
    python evaluation.py --module_path "ensemble"  --class_name "EnsembleModel" --weights_path "weights/" --dataset_path "../../datasets/MVTec" --category "$class_name"
    #python evaluation.py --module_path "dinomaly"  --class_name "DinomalyModel" --weights_path "weights/" --dataset_path "../../datasets/MVTec" --category "$class_name"
done




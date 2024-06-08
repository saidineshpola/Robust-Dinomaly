import os
import shutil

classes = [  # classes except single ones
    'grid', 'leather', 'tile', 'wood',
    'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
]

source_file = 'invad_mvtec_net_180.pth'

for class_name in classes:
    # class_name = f'mvtec_{class_name}'
    # Create a new directory for the class
    os.makedirs(class_name, exist_ok=True)

    # Copy the file to the new directory
    shutil.copy(source_file, os.path.join(class_name, f'{class_name}.pth'))

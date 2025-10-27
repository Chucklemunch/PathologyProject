import zipfile 
import os

# Images include 5 classes
# 5000: Colon Adenocarcinomas (colonca)
# 5000: Benign Colonic Tissues (colonn)
# 5000: Lung Adenocarcinomas (lungaca)
# 5000: Lung Squamous Cell Carcinomas (lungscc)
# 5000: Benign Lung Tissues (lungn)

# import data
img_paths = []
for root, dirs, files in os.walk('lung_colon_image_set/'):
    if len(files) != 0:
        img_paths.extend(files)


# Data

The images are NOT included in this repository.

Use the public PlantVillage dataset (for example the Kaggle dataset `abdallahalidev/plantvillage-dataset`) and point
`--data_dir` at the folder that holds the colour images, one sub-folder per class:

```
data/plantvillage/color/
├── Tomato___Bacterial_spot/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___Leaf_Mold/
├── Tomato___Septoria_leaf_spot/
├── Tomato___Spider_mites Two-spotted_spider_mite/
├── Tomato___Target_Spot/
├── Tomato___Tomato_mosaic_virus/
├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
├── Tomato___healthy/
└── (other crops are ignored; only folders starting with "Tomato" are used)
```

The training script keeps the 10 tomato classes, optionally limits each class to a fixed number of images
(`--max_per_class`), and splits the images 70 % train, 15 % validation and 15 % test (stratified).
Use the `color` folder, not the `grayscale` or `segmented` folders.

Add here: dataset link and licence.

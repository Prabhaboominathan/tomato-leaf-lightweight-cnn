# Lightweight SE + Swish CNN for Tomato Leaf Disease Classification (PyTorch)

An independent PyTorch re-implementation of a compact convolutional network, built from scratch (no pretrained
weights), that classifies tomato leaves into **10 classes**: bacterial spot, early blight, late blight, leaf mold,
septoria leaf spot, two-spotted spider mite, target spot, tomato mosaic virus, yellow leaf curl virus and healthy.
It also includes the ablation runs and the model-size / CPU-latency check used to judge edge suitability.

## Architecture

```
224 x 224 x 3 image, pixels scaled to [0, 1]
  -> stem: 3x3 conv, stride 2, 32 filters, BN, Swish                 112 x 112 x 32
  -> 5 bottleneck blocks, each: 1x1 expand -> 3x3 depthwise
       -> Squeeze-and-Excitation (reduction 1/16) -> 1x1 project      64 / 128 / 256 / 384 / 512 channels
       (skip connection when input and output shapes match)           down to 7 x 7 x 512
  -> global average pooling                                           512
  -> dense 128 + Swish + dropout 0.4
  -> dense 10 (softmax)
```

**Swish** is `x * sigmoid(x)` (PyTorch `SiLU`). **Squeeze-and-Excitation (SE)** re-weights the channels using global
pooling, two small dense layers and a sigmoid gate. Depthwise convolutions keep the parameter count and compute low.

## Dataset

The public PlantVillage dataset, tomato classes only (see `data/README.md`). By default each class is limited to
2000 images and the data are split 70 % train, 15 % validation, 15 % test (stratified).

## Setup and run

```bash
pip install -r requirements.txt
python -m src.train --data_dir data/plantvillage/color --out_dir results
```

Training uses Adam (weight decay 1e-4), cross-entropy loss, light augmentation (rotation 25 degrees, shift 10 %,
zoom 20 %, shear, horizontal flip) and 60 epochs. The epoch with the lowest validation loss is kept.
A GPU is recommended: `notebooks/run_in_colab.ipynb` runs everything on Google Colab.

### Ablation study (does SE help? does Swish help?)

```bash
python -m src.train --data_dir data/plantvillage/color --out_dir results_no_se --no_se
python -m src.train --data_dir data/plantvillage/color --out_dir results_relu  --activation relu
python -m src.compare_runs results results_no_se results_relu --out results/ablation
```

### Model size, speed, prediction and export

```bash
python -m src.benchmark --ckpt results/checkpoints/model.pth        # parameters, size, CPU latency
python -m src.predict --image leaf.jpg --ckpt_dir results/checkpoints
python -m src.export_onnx --ckpt_dir results/checkpoints --out results/model.onnx
```

## Outputs (created in `results/` by the run)

| File | Content |
|---|---|
| `metrics_table.csv / .png` | accuracy, precision, recall, F1, sensitivity, specificity, ROC-AUC, MCC, FPR, FNR, log loss |
| `model_per_class_report.csv / .png` | precision, recall, F1 and support for each class |
| `model_confusion_matrix.png` (+ `_normalized`) | confusion matrix |
| `model_roc_curve.png` | one-vs-rest ROC curves and macro AUC |
| `training_curves.png`, `training_log.csv` | accuracy and loss per epoch |
| `ablation.csv / .png / .md` | full model vs. without SE vs. ReLU (after `compare_runs`) |
| `metrics.json`, `RESULTS.md` | all numbers, parameter count, model size, CPU latency |

## Results

Run the training, then paste the tables from `results/RESULTS.md` here and commit the images in `results/`.

![Training curves](results/training_curves.png)
![Confusion matrix](results/model_confusion_matrix.png)
![ROC curves](results/model_roc_curve.png)

## Differences from the original description

- Implemented in PyTorch (the original used TensorFlow / Keras); TFLite quantisation is replaced by an ONNX export.
- The bottleneck expansion ratio is not stated; 6 is used (`--expand`). Check the printed parameter count.
- Dropout 0.4 after the 128-unit dense layer; learning-rate reduction on plateau is added (optional early stopping).
- Augmentation fills empty pixels with black instead of the nearest pixel.
- The original 20,000-image subset is not published, so `--max_per_class` samples a similar balanced subset.

## Tests

```bash
python -m tests.test_smoke     # data selection, metrics, reports and ablation table; no dataset needed
```

## Repository layout

```
src/models.py        stem, bottleneck + SE blocks, classifier
src/data.py          class folders, balancing, 70/15/15 split
src/engine.py        dataset with augmentation, training loop
src/train.py         training and evaluation pipeline
src/compare_runs.py  ablation table
src/benchmark.py     size and CPU latency
src/predict.py       single-image prediction
src/export_onnx.py   ONNX export
src/reporting.py     tables and figures
tests/               smoke tests
notebooks/           Colab notebook
```

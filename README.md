# AML Project

This repository contains machine learning scripts, sensor datasets, plots, and Arduino sketches for activity/posture classification.

## Project Structure

- `AML_DATASET/`: CSV sensor dataset grouped by posture label.
- `AML_ANN.py`: Artificial neural network classifier.
- `AML_DT.py`: Decision tree classifier.
- `AML_KNN.py`: K-nearest neighbors classifier.
- `AML_SVM.py`: Support vector machine classifier.
- `AML_READ_DATA.py`: Serial data collection script.
- `AML_generate_participants.py`: Synthetic/augmented participant data generation script.
- `plot.py`: Visualization and model comparison plots.
- `draw_confusion_matrix.py`: Confusion matrix plotting script.
- `AML_Device/`: Arduino device sketch.
- `AML_Device_IOT/`: Arduino IoT sketch and exported SVM header.

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run a classifier with its default settings:

```bash
python AML_SVM.py --no_confirm
python AML_KNN.py --no_confirm
python AML_DT.py --no_confirm
python AML_ANN.py --no_confirm
```

Generate plots:

```bash
python plot.py
```

Collect serial data from the device:

```bash
python AML_READ_DATA.py
```

## Dataset

The dataset is stored in `AML_DATASET/`. Each posture folder contains participant CSV files with normalized FSR, ToF, and derived ratio features.


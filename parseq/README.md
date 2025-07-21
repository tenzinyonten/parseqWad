# PARSeq on Tibetan Modern Print Dataset

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

The training dataset is the Tibetan Modern Print Dataset.

Dataset: NorbuKetaka + Google Books

| Split  | Number of Lines |
|--------|----------------|
| Train  | 2,409,802      |
| Val    | 298,940        |
| Test   | 298,244        |

For training the model, the dataset must be stored in LMDB format. We use the following script to generate LMDB files: https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/create_lmdb_dataset.py

Directory structure:
```
Dataset/
└── modern/
    ├── training_lmdb/
    │   ├── real/
    ├── validation_lmdb/
    └── test_lmdb/
```

## Config
The training configuration is specified in `configs/main.yaml`. The Tibetan character set used for training is defined in `configs/charset/tibetan.yaml`.

## Training
With the config file prepared, the model can be trained using the following command:
```bash
python train.py 
```

## Evaluation
The trained checkpoint is saved in the checkpoints/ directory.
For evaluation, use the test.py script with the appropriate arguments.

```bash
python test.py checkpoints/PARSeq-Eric-Split-modern-print_models_parseq_outputs_parseq_2024-12-25_09-18-19_checkpoints_epoch=12-step=471848-val_accuracy=87.7410-val_NED=99.5739.ckpt --data_root Dataset/modern/test_lmdb
```
### Results:

| Dataset      | # Samples | Accuracy | 1 - NED | Confidence | Label Length |
|--------------|-----------|----------|---------|------------|--------------|
| Google Books | 75,164    | 82.49    | 98.81   | 87.43      | 60.46        |
| Norbuketaka  | 223,080   | 89.55    | 99.83   | 93.87      | 82.11        |
| Combined     | 298,244   | 87.77    | 99.57   | 92.25      | 76.66        |



### Inference on single image

```bash
python read.py checkpoints/PARSeq-Eric-Split-modern-print_models_parseq_outputs_parseq_2024-12-25_09-18-19_checkpoints_epoch=12-step=471848-val_accuracy=87.7410-val_NED=99.5739.ckpt --images /path/to/image
```




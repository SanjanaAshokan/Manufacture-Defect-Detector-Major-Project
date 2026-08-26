# MVTec AD Transistor Dataset

## Overview
This project uses the **MVTec Anomaly Detection (AD)** dataset — specifically the **Transistor** category — for binary defect classification.

## Download Instructions

### Option 1: Official Website
1. Visit [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. Accept the license agreement (CC BY-NC-SA 4.0)
3. Download the **transistor** category (~300 MB)
4. Extract to `data/mvtec_transistor/`

### Option 2: Kaggle
1. Visit [MVTec AD on Kaggle](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)
2. Download and extract the transistor folder to `data/mvtec_transistor/`

### Option 3: Automated Script
```bash
python data/download_dataset.py
```

## Expected Raw Structure
```
data/mvtec_transistor/
├── train/
│   └── good/              (213 defect-free images)
├── test/
│   ├── good/              (60 defect-free images)
│   ├── bent_lead/         (defective images)
│   ├── cut_lead/          (defective images)
│   ├── damaged_case/      (defective images)
│   └── misplaced/         (defective images)
└── ground_truth/
    ├── bent_lead/         (binary masks)
    ├── cut_lead/
    ├── damaged_case/
    └── misplaced/
```

## Processed Structure (after running `download_dataset.py`)
```
data/processed/
├── train/
│   ├── good/
│   └── defective/
├── val/
│   ├── good/
│   └── defective/
└── test/
    ├── good/
    └── defective/
```

## License
The MVTec AD dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — non-commercial use only.

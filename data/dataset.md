# COCO 2017 dataset layout and setup

Place the COCO 2017 data under `data/` with this structure:

```
data/
├── train2017/                       # 118,287 training images
├── val2017/                         # 5,000 validation images
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

## Download (Bash)
```bash
mkdir -p data && cd data
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
```
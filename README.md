# Intelligent Frequency-space Image Filtering for Computer Vision Problems in Medicine

## Installation Requirements

TBD

## Datasets. How to Download and Structure

* [Breast Ultrasound Images Dataset (Dataset **BUSI**) (253 MB)](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)

  Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
  
  The archive should be unpacked and turned into the following structure:

  ```
  ├───BUSI
      ├───benign
          ├───benign (1).png
          ├───benign (1)_mask.png
          └─── ...
      ├───malignant
          ├───malignant (1).png
          ├───malignant (1)_mask.png
          └─── ...
      ├───normal
          ├───normal (1).png
          ├───normal (1)_mask.png
          └─── ...
  ```

* [Brachial Plexus Ultrasound Images (**BPUI**) train (1.08 GB)](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data)

  The archive should be unpacked and turned into the following structure:

    ```
    ├───BPUI
        ├───1_1.tif
        ├───1_1_mask.tif
        └─── ...
    ```

## How to Train Models

To train a model, you should 

1. initially choose a CV task:
    * Segmentation
    * Classification
    * Denoising (Erasing)

2. correct [`configs.yaml`](https://github.com/vitekspeedcuber/Ultrasound/blob/main/configs.yaml) file

3. run appropriate script `.py`:

```
python ./train_<CV task>.py
```

or with parameters
  * --device &emsp; &emsp; &emsp; # cuda device id number (default 0)
  * --random_seed &nbsp; # random seed (default 1)
  * --nolog &emsp; &emsp; &emsp; # turn off logging

## Train Process Performance

As a result, you will receive the saved best model `.pth` and a log file `.tfevents` of the training process next to files with code.

# Intelligent Frequency-space Image Filtering for Computer Vision Problems in Medicine

## Installation Requirements

The file [`environment.yml`](https://github.com/vitekspeedcuber/Ultrasound/blob/main/environment.yml) contains the necessary libraries to run all scripts

It is necessary:

1. create conda environment and install all packages through the command: `conda env create -f environment.yml`
2. activate environment to run scripts from it: `conda activate ultrasound`

## Datasets. How to Download and Structure

* **Endocrinology** Dataset
  
  This dataset provided by [the Center for Endocrinology](https://www.endocrincentr.ru/) and is not in the public domain.

* [Breast Ultrasound Images Dataset (Dataset **BUSI**) (253 MB)](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)

  `Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.`
  
  The archive should be unpacked and turned into the following structure:

  ```
  ├───BUSI_data
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
    ├───BPUI_train
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

   ! Do not forget to specify **path to folder with train data**.
   
   ! For classification task use `number of classes = 3` for **BUSI** dataset and `number of classes = 6` for **Endocrinology** dataset.

3. for denoising/erasing task comment/uncomment lines in file `train_denoising.py` with appropriate noise overlaying transform

4. run appropriate script `.py`:

```
python ./train_<CV task>.py
```

or with parameters
  * --device &emsp; &emsp; &emsp; # cuda device id number (default 0)
  * --random_seed &nbsp; # random seed (default 1)
  * --nolog &emsp; &emsp; &emsp; # turn off logging

## Train Process Performance

As a result, you will receive 

* the saved best model `.pth` next to files with code
* a log file `.tfevents` of the training process in folder where the results of experiments from the tensorboard will be recorded (see [`configs.yaml`](https://github.com/vitekspeedcuber/Ultrasound/blob/main/configs.yaml) file)

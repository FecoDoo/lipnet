# Facial Expression Recognition in the Presence of Speech

> A deep neural network that detects emotions through facial expressiosn from talking faces.

## Datasets

- [GRID corpus dataset](https://zenodo.org/record/3625687/)
- [RAVDESS](https://zenodo.org/record/1188976#.YFZuJ0j7SL8)
- [AffectNet](http://mohammadmahoor.com/affectnet/)
- [Oulu-CASIA](https://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engineering/center-machine-vision-and-signal-analysis)
- NBE Datasets (not publically available yet)

## Usage

The DNN consists of two sub-models:
- [LipNet](https://github.com/osalinasv/lipnet)
- [CNN](https://github.com/HSE-asavchenko/face-emotion-recognition)

All scripts for data preperation, model training, and model benchmarks are provided under the directory of `script`.

### Dataset Preparation

Store the dataset under the `data` directory. Each dataset folder should has a structure as follows:

#### Audiovisual Dataset
```bash
dataset
    npy
    videos
```

#### GRID
The GRID dataset requirs additional alignment files (check the LipNet repo for more details), the directory tree should looks like this:

```bash
GRID
    align
    npy
    videos
```


#### Canonical Image Dataset:
```bash
dataset
    train
        faces
            angry
                xxx.jpg
            disgust
            fear
            happy
            neutral
            sad
            surprise
    test
        faces
            ...
```

#### Important Settings
- All images should be stored in the format of `.jpg`.
- All videos should be stored in the format of `.npy`.
- Each face image should be cropped to the shape (224,224,3), and each lip image should be in the shape of (50,100,3). The order of shape is `Width x Height x Channel`.

### Model Training

#### LipNet

Use the `script/lipnet.py` script to start training the lipnet model after preprocessed the GRID dataset.

#### Baseline

Use the `script/baseline.py` script to start training the baseline AFER model (still image based) after preprocessed the affectnet dataset.

#### DNN

Use the `script/dnn.py` script to start training the dnn model after preprocessed the RAVDESS dataset.

### Model Predicting

Use the `predict.py` script to analyze a video or a directory of videos with a trained model:

### Model Evaluation

Scripts under the directory of `script/evaluating` are used for real-time handy evaluation, results are not guaranteed.

## To-do List

- [X] Tensorflow Dataset pipeline
- [X] Generate dummy cropped image representing netual emotion
- [ ] Documentation: Proper usage and code documentation
- [ ] Testing: Develop unit testing

## Author

* **Kai Yao** - [fecodoo](https://github.com/fecodoo) @ Aalto University - Department of Computer Science

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details

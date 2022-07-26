# LipNet

> A Keras implementation of LipNet

This is an implementation of the spatiotemporal convolutional neural network described by Assael et al. in [this article](https://arxiv.org/abs/1611.01599). However, this implementation only tests the unseen speakers task, the overlapped speakers task is yet to be implemented.

The best training completed yet was started the 26th of September, 2018:

|        Task       | Epochs |  CER   |  WER   |
|:-----------------:|:------:|:------:|:------:|
|  Unseen speakers  |   70   |  9.3%  | 15.7%  |

## Usage

### Preprocessing

This project was trained using the [GRID corpus dataset](https://zenodo.org/record/3625687/) as per the original article.

Given the following directory structure:

```
GRID:
├───s1
│   ├───bbaf2n.mpg
│   ├───bbaf3s.mpg
│   └───...
├───s2
│   └───...
└───...
    └───...
```

Use the `preprocessing.py` script to process all videos into `.npy` binary files if the extracted lips. By default, each file has a numpy array of shape (75, 50, 100, 3). That is 75 frames each with 100 pixels in width and 50 in height with 3 channels per pixel.


This results in a new directory with the preprocessed dataset:

```
dataset:
├───s1
├───s2
└───...
```

The original article excluded speakers S1, S2, S20 and S22 from the training dataset.

### Training

Use the `train.py` script to start training a model after preprocesing your dataset. You'll also need to provide a directory containing individual align files with the expected sentence:

### Evaluating

Use the `predict.py` script to analyze a video or a directory of videos with a trained model:


### Configuration

The `env.py` file hosts a number of configurable variables:

Related to the videos:
- **FRAME_COUNT:** The number of frames to be expected for each video
- **IMAGE_WIDTH:** The width in pixels for each video frame
- **IMAGE_HEIGHT:** The height in pixels for each video frame
- **IMAGE_CHANNELS:** The amount of channels for each pixel (3 is RGB and 1 is greyscale)

Related to the neural net:
- **MAX_STRING:** The maximum amount of characters to expect as the encoded align sentence vector
- **OUTPUT_SIZE:** The maximum amount of characters to expect as the prediction output
- **BATCH_SIZE:** The amount of videos to read by batch
- **VAL_SPLIT:** The fraction between 0.0 and 1.0 of the videos to take as the validation set
- **EPOCH:** The number of epochs

Related to the standardization:
- **MEAN_R:** Arithmetic mean of the red channel in the training set
- **MEAN_G:** Arithmetic mean of the green channel in the training set
- **MEAN_B:** Arithmetic mean of the blue channel in the training set
- **STD_R:** Standard deviation of the red channel in the training set
- **STD_G:** Standard deviation of the green channel in the training set
- **STD_B:** Standard deviation of the blue channel in the training set

Related to the decoder:
- **DECODER_GREEDY:** Bool value controls whether to use greedy decoding or not
- **DECODER_BEAM_WIDTH:** Int value defines the width the ctc beam decoding

# preprocessing & training
- **USE_CACHE:**: Bool value that controls cache building
- **VIDEO_SUFFIX:** Suffix of video files, i.e. `.mp4`

# prediction
DICTIONARY_PATH = "data/dictionaries/grid.txt"
MODEL_PATH = "models/lipnet.h5"
VIDEO_PATH = "videos"
DLIB_SHAPE_PREDICTOR_PATH = "data/dlib/shape_predictor_68_face_landmarks.dat"

# others
TF_CPP_MIN_LOG_LEVEL = "3"

## To-do List

- [ ] Tensorflow Dataset pipeline
- [ ] Generate dummy cropped image representing netual emotion
- [ ] Documentation: Proper usage and code documentation
- [ ] Testing: Develop unit testing

## Author

* **Omar Salinas** - [omarsalinas16](https://github.com/omarsalinas16) Developed as a bachelor's thesis @ UACJ - IIT
* **Kai Yao** - [fecodoo](https://github.com/fecodoo) @ Aalto University - Department of Computer Science

See also the list of [contributors](https://github.com/omarsalinas16/lipnet/contributors) who participated in this project.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details

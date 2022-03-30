FRAME_COUNT = 75
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 50
IMAGE_CHANNELS = 3

MAX_STRING = 32
OUTPUT_SIZE = 28

BATCH_SIZE = 16
VAL_SPLIT = 0.2

MEAN_R = 0.6882
MEAN_G = 0.5057
MEAN_B = 0.3344

STD_R = 0.1629
STD_G = 0.1205
STD_B = 0.1257

# preprocessing & training
DEV = 0
USE_CACHE = 1
EPOCH = 60

VIDEO_PATTERN = "*.mpg"

# prediction
DICTIONARY_PATH = "data/dictionaries/grid.txt"
MODEL_PATH = "models/lipnet.h5"
VIDEO_PATH = "videos"
DLIB_SHAPE_PREDICTOR_PATH = "data/dlib/shape_predictor_68_face_landmarks.dat"

# others
TF_CPP_MIN_LOG_LEVEL = "3"

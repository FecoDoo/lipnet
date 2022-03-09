import argparse
import datetime
import os
import time
from typing import NamedTuple

from colorama import Fore, init
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

import env
from common.decode import create_decoder
from core.callbacks.error_rates import ErrorRates
from core.generators.dataset_generator import DatasetGenerator
from core.model.lipnet import LipNet
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
init(autoreset=True)


ROOT_PATH  = Path(os.path.dirname(os.path.realpath(__file__))).resolve()
OUTPUT_DIR = Path(os.path.realpath(os.path.join(ROOT_PATH, 'data', 'res'))).resolve()
LOG_DIR    = Path(os.path.realpath(os.path.join(ROOT_PATH, 'data', 'res_logs'))).resolve()

DICTIONARY_PATH = Path(os.path.realpath(ROOT_PATH.join(os.path.join('data', 'dictionaries', 'grid.txt')))).resolve()


class TrainingConfig(NamedTuple):
	dataset_path:   str
	aligns_path:    str
	epochs:         int = 1
	frame_count:    int = env.FRAME_COUNT
	image_width:    int = env.IMAGE_WIDTH
	image_height:   int = env.IMAGE_HEIGHT
	image_channels: int = env.IMAGE_CHANNELS
	max_string:     int = env.MAX_STRING
	batch_size:     int = env.BATCH_SIZE
	val_split:    float = env.VAL_SPLIT
	use_cache:     bool = True


def main():
	"""
	Entry point of the script for training a model.
	i.e: python train.py -d data/dataset -a data/aligns -e 150
	"""

	print(r'''
   __         __     ______   __   __     ______     ______  
  /\ \       /\ \   /\  == \ /\ "-.\ \   /\  ___\   /\__  _\ 
  \ \ \____  \ \ \  \ \  _-/ \ \ \-.  \  \ \  __\   \/_/\ \/ 
   \ \_____\  \ \_\  \ \_\    \ \_\\"\_\  \ \_____\    \ \_\ 
    \/_____/   \/_/   \/_/     \/_/ \/_/   \/_____/     \/_/ 

  implemented by Omar Salinas
	''')

	ap = argparse.ArgumentParser()

	ap.add_argument('-d', '--dataset-path', required=True, help='Path to the dataset root directory')
	ap.add_argument('-a', '--aligns-path', required=True, help='Path to the directory containing all align files')
	ap.add_argument('-e', '--epochs', required=False, help='(Optional) Number of epochs to run', type=int, default=1)
	ap.add_argument('-ic', '--ignore-cache', required=False, help='(Optional) Force the generator to ignore the cache file', action='store_true', default=False)

	args = vars(ap.parse_args())

	dataset_path = os.path.realpath(args['dataset_path'])
	aligns_path  = os.path.realpath(args['aligns_path'])
	epochs       = args['epochs']
	ignore_cache = args['ignore_cache']

	if not dataset_path.is_dir():
		print(Fore.RED + '\nERROR: The dataset path is not a directory')
		return

	if not aligns_path.is_dir():
		print(Fore.RED + '\nERROR: The aligns path is not a directory')
		return

	if not isinstance(epochs, int) or epochs <= 0:
		print(Fore.RED + '\nERROR: The number of epochs must be a valid integer greater than zero')
		return

	name   = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
	config = TrainingConfig(dataset_path, aligns_path, epochs=epochs, use_cache=not ignore_cache)

	train(name, config)


def train(run_name: str, config: TrainingConfig):
	print("\nTRAINING: {}\n".format(run_name))

	print('For dataset at: {}'.format(config.dataset_path))
	print('With aligns at: {}'.format(config.aligns_path))

	if not OUTPUT_DIR.exists():
		OUTPUT_DIR.mkdir() 
	if not LOG_DIR.exists():
		LOG_DIR.mkdir()

	lipnet = LipNet(config.frame_count, config.image_channels, config.image_height, config.image_width, config.max_string).compile_model()

	datagen = DatasetGenerator(config.dataset_path, config.aligns_path, config.batch_size, config.max_string, config.val_split, config.use_cache)

	callbacks = create_callbacks(run_name, lipnet, datagen)

	print('\nStarting training...\n')

	start_time = time.time()

	lipnet.model.fit_generator(
		generator      =datagen.train_generator,
		validation_data=datagen.val_generator,
		epochs         =config.epochs,
		verbose        =1,
		shuffle        =True,
		max_queue_size =5,
		workers        =2,
		callbacks      =callbacks,
		use_multiprocessing=True
	)

	elapsed_time = time.time() - start_time
	print('\nTraining completed in: {}'.format(datetime.timedelta(seconds=elapsed_time)))


def create_callbacks(run_name: str, lipnet: LipNet, datagen: DatasetGenerator) -> list:
	run_log_dir = LOG_DIR.joinpath(run_name)
	
	if not run_log_dir.exists():
		run_log_dir.mkdir()

	# Tensorboard
	tensorboard = TensorBoard(log_dir=str(run_log_dir))

	# Training logger
	csv_log    = run_log_dir.joinpath('training.csv')
	csv_logger = CSVLogger(csv_log, separator=',', append=True)

	# Model checkpoint saver
	checkpoint_dir = OUTPUT_DIR.joinpath(run_name)
	if not checkpoint_dir.exists():
		checkpoint_dir.mkdir()

	checkpoint_template = os.path.join(checkpoint_dir, "lipnet_{epoch:03d}_{val_loss:.2f}.hdf5")
	checkpoint = ModelCheckpoint(checkpoint_template, monitor='val_loss', save_weights_only=True, mode='auto', period=1, verbose=1)

	# WER/CER Error rate calculator
	error_rate_log = os.path.join(run_log_dir, 'error_rates.csv')

	decoder = create_decoder(DICTIONARY_PATH, False)
	error_rates = ErrorRates(error_rate_log, lipnet, datagen.val_generator, decoder)

	return [checkpoint, csv_logger, error_rates, tensorboard]


if __name__ == '__main__':
	main()

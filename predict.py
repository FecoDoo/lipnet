import argparse
import dlib
import env
import numpy as np
import os
import skvideo.io

from colorama import init, Fore
from common.files import is_dir, is_file, get_file_extension, get_files_in_dir, walk_level
from lipnext.helpers.video import reshape_and_normalize_video_data
from preprocessing.extractor.extract_roi import extract_video_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
init(autoreset=True)


ROOT_PATH          = os.path.dirname(os.path.realpath(__file__))
DICTIONARY_PATH    = os.path.realpath(os.path.join(ROOT_PATH, 'data', 'dictionaries', 'grid.txt'))
DECODER_GREEDY     = True
DECODER_BEAM_WIDTH = 200


# set PYTHONPATH=%PYTHONPATH%;./
# python predict.py -w data\results\2018-07-31-20-09-20\w_0115_2.19.hdf5 -v D:\GRID\s1\bbaf2n.mpg
# bin blue at f two now
def predict(weights_file_path: str, video_path: str, predictor_path: str, frame_count: int, image_width: int, image_height: int, image_channels: int, max_string: int):
	from lipnext.decoding.decoder import Decoder
	from lipnext.decoding.spell import Spell
	from lipnext.model.v4 import Lipnext
	from lipnext.utils.labels import labels_to_text
	
	
	print("\nPREDICTION\n")

	video_path_is_file = is_file(video_path) and not is_dir(video_path)

	if video_path_is_file:
		print('Predicting for video at: {}'.format(video_path))
		video_paths = [video_path]
	else:
		print('Predicting batch at:     {}'.format(video_path))
		video_paths = get_video_files_in_dir(video_path)

	print('Loading weights at:      {}'.format(weights_file_path))
	print('Using predictor at:      {}\n'.format(predictor_path))

	detector  = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	print('\nExtracting input video data...')

	input_data = list(map(lambda x: (x, path_to_video_data(x, detector, predictor)), video_paths))
	input_data = list(filter(lambda x: x[1] is not None, input_data))

	if len(input_data) <= 0:
		print(Fore.RED + '\nNo valid video files were found, exiting.')
		return

	print('\nMaking predictions...')

	lipnext = Lipnext(frame_count, image_channels, image_height, image_width, max_string)
	lipnext.compile_model()

	lipnext.model.load_weights(weights_file_path)

	x_data = np.array([x[1] for x in input_data])
	y_pred = lipnext.predict(x_data)

	spell   = Spell(DICTIONARY_PATH)
	decoder = Decoder(greedy=DECODER_GREEDY, beam_width=DECODER_BEAM_WIDTH, postprocessors=[labels_to_text, spell.sentence])

	input_length = np.array([len(x) for x in x_data])
	results = decoder.decode(y_pred, input_length)

	print('\n\nRESULTS:')

	for (i, _), s, r in zip(input_data, y_pred, results):
		print('\nVideo: {}\n   Shape:  {}\n   Result: {}'.format(i, s.shape, r))


def get_video_files_in_dir(path: str) -> [str]:
	video_paths = []

	for sub_dir, _, _ in walk_level(path):
		video_paths += [f for f in get_files_in_dir(sub_dir, '*.mpg')]

	return video_paths


def path_to_video_data(path: str, detector, predictor) -> np.ndarray:
	data = extract_video_data(path, detector, predictor)

	if data is not None:
		data = reshape_and_normalize_video_data(data)
		return data
	else:
		return None


def main():
	print(r'''
   __         __     ______   __   __     ______     __  __     ______  
  /\ \       /\ \   /\  == \ /\ "-.\ \   /\  ___\   /\_\_\_\   /\__  _\ 
  \ \ \____  \ \ \  \ \  _-/ \ \ \-.  \  \ \  __\   \/_/\_\/_  \/_/\ \/ 
   \ \_____\  \ \_\  \ \_\    \ \_\\"\_\  \ \_____\   /\_\/\_\    \ \_\ 
    \/_____/   \/_/   \/_/     \/_/ \/_/   \/_____/   \/_/\/_/     \/_/ 
	''')

	ap = argparse.ArgumentParser()

	ap.add_argument('-v', '--video-path', required=True,
		help='Path to video file or batch directory to analize')

	ap.add_argument('-w', '--weights-path', required=True,
		help='Path to .hdf5 trained weights file')

	DEFAULT_PREDICTOR = os.path.join(__file__, '..', 'data', 'predictors', 'shape_predictor_68_face_landmarks.dat')

	ap.add_argument("-pp", "--predictor-path", required=False,
		help="(Optional) Path to the predictor .dat file", default=DEFAULT_PREDICTOR)

	args = vars(ap.parse_args())

	weights        = os.path.realpath(args['weights_path'])
	video          = os.path.realpath(args['video_path'])
	predictor_path = os.path.realpath(args["predictor_path"])

	if not is_file(weights) or get_file_extension(weights) != '.hdf5':
		print(Fore.RED + '\nERROR: Trained weights path is not a valid file')
		return

	if not is_file(video) and not is_dir(video):
		print(Fore.RED + '\nERROR: Path does not point to a video file nor to a directory')
		return

	if not is_file(predictor_path) or get_file_extension(predictor_path) != '.dat':
		print(Fore.RED + '\nERROR: Predictor path is not a valid file')
		return
	
	predict(weights, video, predictor_path, env.FRAME_COUNT, env.IMAGE_WIDTH, env.IMAGE_HEIGHT, env.IMAGE_CHANNELS, env.MAX_STRING)


if __name__ == '__main__':
	main()

import datetime
import os
import pickle
import time

from loggingUtils import error

SAVE_FOLDER_NAME = "saves"
TIME_FORMAT = '%d-%m-%Y_%H:%M:%S'


class ObjectMapper(object):
	def __init__(self, ext):
		self.ext = ext

	def save(self, model=None):
		if model is None:
			model = get_time()

		if not os.path.exists(SAVE_FOLDER_NAME):
			os.mkdir(SAVE_FOLDER_NAME)

		dump_file_name = '%s.%s' % (model, self.ext)
		dump_file = open(os.path.join(SAVE_FOLDER_NAME, dump_file_name), 'w')
		pickle.dump(self, dump_file)


def load(prefix, ext):
	if not os.path.exists(SAVE_FOLDER_NAME):
		os.mkdir(SAVE_FOLDER_NAME)
	dump_file_name = '%s.%s' % (prefix, ext)
	dump_file_path = os.path.join(SAVE_FOLDER_NAME, dump_file_name)
	if not os.path.isfile(dump_file_path):
		error('Failed to load saved file %s. File does not exist.' % dump_file_name)
		return None
	return pickle.load(open(dump_file_path, 'r'))


def get_time():
	return datetime.datetime.fromtimestamp(time.time()).strftime(TIME_FORMAT)

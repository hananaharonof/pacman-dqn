import inspect
import time
from termcolor import colored

INFO = "INFO"
DEBUG = "DEBUG"
ERROR = "ERROR"


def info(msg):
	log(msg, INFO)


def debug(msg):
	log(msg, DEBUG, 'yellow')


def error(msg):
	log(msg, ERROR, 'red')


def log(msg, level, color='green'):
	localtime = time.asctime(time.localtime(time.time()))
	print colored("[%s] [%s] [%s] %s" % (localtime, level, caller_name(), msg), color)


def caller_name():
	curr_frame = inspect.currentframe()
	call_frame = inspect.getouterframes(curr_frame, 2)
	return "%s:%s" % (call_frame[3][3], call_frame[3][2])

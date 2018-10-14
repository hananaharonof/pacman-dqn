import inspect
import time
from termcolor import colored

INFO = "INFO"
DEBUG = "DEBUG"
ERROR = "ERROR"


def info(msg, caller=None):
	log(msg, INFO, caller)


def debug(msg, caller=None):
	log(msg, DEBUG, caller, 'yellow')


def error(msg, caller=None):
	log(msg, ERROR, caller, 'red')


def log(msg, level, caller=None, color='green'):
	if caller is None:
		caller = caller_name()
	localtime = time.asctime(time.localtime(time.time()))
	print colored("[%s] [%s] [%s] %s" % (localtime, level, caller, msg), color)


def caller_name():
	curr_frame = inspect.currentframe()
	call_frame = inspect.getouterframes(curr_frame, 2)
	return "%s:%s" % (call_frame[3][3], call_frame[3][2])

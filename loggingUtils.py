import inspect
import time

INFO = "INFO"


def info(msg):
	log(msg, INFO)


def log(msg, level):
	localtime = time.asctime(time.localtime(time.time()))
	print "[%s] [%s] [%s] %s" % (localtime, level, caller_name(), msg)


def caller_name():
	curr_frame = inspect.currentframe()
	call_frame = inspect.getouterframes(curr_frame, 2)
	return "%s:%s" % (call_frame[3][3], call_frame[3][2])




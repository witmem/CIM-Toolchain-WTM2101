from curses.ascii import NUL
import os
import logging
# from turtle import position
import tvm as witin
import tvm.relay as witin_frontend
import sys
import datetime

''' log level set
 0: print log for level fatal/error/warning
 1: print log for level fatal/error/warning/information
 2: print log for level fatal/error/warning/information/debug
'''
os.environ['DMLC_LOG_DEBUG'] = '0'
# save log to file
os.environ['SAVE_LOG_MESSAGE'] = '0'

# The first N rows to retain array space are not allocated
os.environ['ARRAY_ALLOC_RESERVED_COLUMN'] = '128'
# Open the fifo in Regfile
os.environ['WITIN_FIFO_EN'] = '0'
# Compensation calibration data selected average frame
# Otherwise(default), select the most similar to the average frame
# os.environ['WITIN_ADDITIONAL_DATA_USE_AVG'] = '0' # Discard
# Burns Array the number of selected frames
os.environ['WITIN_EXPECTED_NUM'] = '100'

########################## not change ##########################
# witin_mapper version
os.environ['WITIN_MAPPER_VERSION'] = "v001.003.000"
# tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

path_position = os.path.abspath(sys.argv[0]).find("witin_mapper")
relative_path = " "
if path_position != -1:
    relative_path = os.path.abspath(sys.argv[0])[path_position:]
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s " + relative_path + ":" + "%(message)s "  # config output format
DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a '  # date
LOG_NAME_DATE = datetime.datetime.now().strftime('%H:%M')
LOG_SAVE_ROOT_PATH = "../witin_mapper_log_txt_"
if os.environ['SAVE_LOG_MESSAGE'] == '1':
    logging.basicConfig(level=logging.ERROR,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        filename=LOG_SAVE_ROOT_PATH + LOG_NAME_DATE + ".txt")
else:
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT, datefmt=DATE_FORMAT)

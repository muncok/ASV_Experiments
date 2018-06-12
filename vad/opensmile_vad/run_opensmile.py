# Run opensmile voice detection for the whole data
import os
import time
import multiprocessing as mp
import pickle

OPENSMILE_PATH = './SMILExtract'
CONF_FILE_PATH = 'vad_segmenter.conf'
DATA_PATH = '/home/muncok/DL/dataset/SV_sets/reddots_r2015q4_v1/wav/'

#SMILE_CMD = '%s -C %s -I %s -O smile.csv' % (OPENSMILE_PATH, CONF_FILE_PATH, TEST_FILE_PATH)
SMILE_CMD = '%s -C %s -I %s -waveoutput %s 2> /dev/null'
# SMILE_CMD = '%s -C %s -I %s -waveoutput %s'

# fs = os.listdir(DATA_PATH)
fs = pickle.load(open("reddots_files.pkl", "rb"))
ix = mp.Value('i', 0)

def process_file(f):
  ix.value += 1
  print(ix.value, f)
  ff = DATA_PATH + f
  output_wav = 'reddots_vad/%s' % f.split('/')[-1]
  cmd = SMILE_CMD % (OPENSMILE_PATH, CONF_FILE_PATH, ff, output_wav)
  os.system(cmd)

pool = mp.Pool(processes = 150)
pool.map(process_file, fs)

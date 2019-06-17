import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--score_file', type=str)
parser.add_argument('--output_file', type=str)
args = parser.parse_args()

scores = pd.read_csv(args.score_file, sep=" ", names=['enr', 'test', 'score'])
scores['test'] = scores.test.apply(lambda x: 'sid_eval/'+x+'.wav')
scores.to_csv(args.output_file, sep=" ", header=False, index=False)

os.system("./score_voices -c {} {}".format(args.output_file,
"voices/eval_set/sid_eval_lists/eval-trial.lst"))


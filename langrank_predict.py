import langrank as lr
import os
import argparse


parser = argparse.ArgumentParser(description='Langrank parser.')
parser.add_argument('-o', '--orig', type=str, required=True, help='unsegmented dataset')
parser.add_argument('-s', '--seg', type=str, help='segmented dataset')
parser.add_argument('-l', '--lang', type=str, required=True, help='language code')
parser.add_argument('-n', '--num', type=int, default=3, help='print top N')

params = parser.parse_args()

assert os.path.isfile(params.orig)
assert os.path.isfile(params.seg) or params.seg is None

with open(params.orig) as inp:
	lines = inp.readlines()

bpelines = None
if params.seg is not None:
	with open(params.seg) as inp:
		bpelines = inp.readlines()

print("read lines")
prepared = lr.prepare_new_dataset(params.lang, dataset_source=lines, dataset_subword_source=bpelines)
print("prepared")
lr.rank(prepared, candidates="all", print_topK=params.num)
print("ranked")


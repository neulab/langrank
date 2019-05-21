import langrank as lr
import sys

lang=sys.argv[1]
dataset = "sample-data/ted-train.orig.{}".format(lang)
bpe_dataset = "sample-data/ted-train.orig.spm8000.{}".format(lang)
with open(dataset) as inp:
	lines = inp.readlines()
with open(bpe_dataset) as inp:
	bpelines = inp.readlines()


print("read lines")
prepared = lr.prepare_new_dataset(lang, dataset_source=lines, dataset_subword_source=bpelines)
print("prepared")
# lr.rank(prepared, candidates=['ell', 'ara', 'slv', 'slk', 'ita'])
lr.rank(prepared, model=lang, candidates=["-{}".format(lang)])
print("ranked")


import langrank as lr

with open("sample-data/ell.tok") as inp:
	lines = inp.readlines()
with open("sample-data/ell.tok.bpe") as inp:
	bpelines = inp.readlines()


print("read lines")
prepared = lr.prepare_new_dataset('ell', dataset_source=lines, dataset_subword_source=bpelines)
print("prepared")
lr.rank(prepared, candidates=['ell', 'ara', 'slv', 'slk', 'ita'])
print("ranked")


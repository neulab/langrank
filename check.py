import langrank as lr

with open("sample-data/sl.tok") as inp:
	lines = inp.readlines()

print("read lines")
prepared = lr.prepare_new_dataset('slv', dataset_source=lines)
print("prepared")
lr.rank(prepared, candidates=['ell', 'ara', 'slv', 'slk'])
print("ranked")


import os
import subprocess
import numpy as np


def read_data(fn):
	with open(fn) as inp:
		lines = inp.readlines()
	c = []
	v = []
	for l in lines:
		l = l.strip().split()
		if len(l) == 2:
			c.append(int(l[1]))
			v.append(l[0])
	return v,c


dataset_dir = "datasets"

# Not needed now
#eng_vocab_f = "datasets/eng/word.vocab"
#en_v, en_c = read_data(eng_vocab_f)
#eng_vocab_f = "datasets/eng/subword.vocab"
#en_sub_v, en_sub_c = read_data(eng_vocab_f)

#w2i = {w:i for i,w in enumerate(en_v)}
#subw2i = {w:i for i,w in enumerate(en_sub_v)}


features = {}
# Add if adding target-side features for MT
#features["eng"] = {}
#features["eng"]["word_vocab"] = en_v
#features["eng"]["subword_vocab"] = en_sub_v

for directory in os.listdir(dataset_dir):
	if directory != "eng":
	#if directory == "ell_eng":
		# Get number of lines in training data
		temp = directory.split("_")
		filename = "ted-train.orig."+temp[0]
		language = temp[0]
		if len(language) != 3:
			continue
		filename = os.path.join("datasets",directory,filename)
		bashCommand = "wc -l " + filename
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()
		lines = int(output.strip().split()[0])
		print(filename + " " + str(lines))

		key = "ted_"+temp[0]
		features[key] = {}
		features[key]["lang"] = language
		features[key]["dataset_size"] = lines

		# Get number of types and tokens
		filename = "word.vocab"
		filename = os.path.join("datasets",directory,filename)
		vocab, counts = read_data(filename)
		features[key]["token_number"] = np.sum(counts)
		features[key]["type_number"] = len(vocab)
		features[key]["word_vocab"] = vocab
		features[key]["type_token_ratio"] = features[key]["type_number"]/float(features[key]["token_number"])


		# Get number of subword types and tokens
		filename = "subword.vocab"
		filename = os.path.join("datasets",directory,filename)
		vocab, counts = read_data(filename)
		features[key]["subword_token_number"] = np.sum(counts)
		features[key]["subword_type_number"] = len(vocab)
		features[key]["subword_vocab"] = vocab
		features[key]["subword_type_token_ratio"] = features[key]["subword_type_number"]/float(features[key]["subword_token_number"])

		'''
		# Get number of types and tokens for English side
		filename = "en.word.vocab"
		filename = os.path.join("datasets",directory,filename)
		vocab, counts = read_data(filename)
		features[key]["en_token_number"] = np.sum(counts)
		features[key]["en_type_number"] = len(vocab)
		v = np.zeros(len(w2i))
		for w,c in zip(vocab,counts):
			v[w2i[w]] = c
		features[key]["en_word_vocab"] = v
		features[key]["en_type_token_ratio"] = features[key]["en_type_number"]/float(features[key]["en_token_number"])


		# Get number of subword types and tokens
		filename = "en.subword.vocab"
		filename = os.path.join("datasets",directory,filename)
		vocab, counts = read_data(filename)
		features[key]["en_subword_token_number"] = np.sum(counts)
		features[key]["en_subword_type_number"] = len(vocab)
		v = np.zeros(len(subw2i))
		for w,c in zip(vocab,counts):
			v[subw2i[w]] = c
		features[key]["en_subword_vocab"] = v
		features[key]["en_subword_type_token_ratio"] = features[key]["en_subword_type_number"]/float(features[key]["en_subword_token_number"])
		'''


indexed = "indexed/MT"
outputfile = os.path.join(indexed, "ted.npy")
np.save(outputfile, features)

		







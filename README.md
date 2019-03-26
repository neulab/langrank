# langrank

Steps to install:

First install the latest version of lang2vec:

~~~~
# Install latest lang2vec
git clone https://github.com/antonisa/lang2vec.git
cd lang2vec
wget http://www.cs.cmu.edu/~aanastas/files/distances.npz .
mv distances.npz lang2vec/data/
python3 setup.py install

# Now clone langrank (future: install it as module)
cd ../
git clone https://github.com/antonisa/langrank.git
wget http://www.cs.cmu.edu/~aanastas/files/indexed.tar.gz .
tar -xzvf indexed.tar.gz
~~~~

Further setup (future: provide the pretrained models through wget)
~~~~
mkdir -p pretrained/MT
# copy pretrained .lgbm file there
~~~~

See example in ``check.py`` (it should work if you provide a dataset in your computer).
It follows the example below (ran in the ``langrank`` directory):
~~~~
python3
>>> import langrank as lr
>>>
>>> # Load some data
>>> with open("sample-data/sl.tok") as inp:
...     lines = inp.readlines()
... 
>>> # Just to show that we loaded something
>>> len(lines) 
17022
>>> lines[0]
'Vprašanje je torej , kaj je nevidno ?\n'
>>> 
>>> # Now prepare the dataset
>>> prepared = lr.prepare_new_dataset('slv', dataset_source=lines)
NOTE: no subword-level dataset provided, will only extract word-level features.
>>>
>>> # And rank the candidates (this could be set to 'all' so that it would rank all available datasets)
>>> lr.rank(prepared, candidates=['ell','ara', 'aze'])
ted_ara
[2.76440389e-03 3.76911152e-03 2.14111000e+05 1.70220000e+04
 1.25784867e+01 5.60515862e-02 1.26270588e-01 3.09246574e-01
 6.77100000e-01 1.00000000e+00 1.00000000e+00 6.69000000e-01
 2.00000000e-04 6.41000000e-01]
*****
ted_aze
[1.10159341e-02 8.92949634e-03 5.94600000e+03 1.70220000e+04
 3.49312654e-01 2.21129559e-01 1.26270588e-01 5.64355049e-01
 6.77100000e-01 1.00000000e+00 1.00000000e+00 6.69000000e-01
 2.00000000e-04 6.41000000e-01]
*****
ted_ell
[7.39632260e-03 5.29807427e-03 1.34327000e+05 1.70220000e+04
 7.89137587e+00 4.08389654e-02 1.26270588e-01 4.57754797e-01
 5.34300000e-01 8.33300000e-01 4.56000000e-02 4.41600000e-01
 5.92200000e-01 5.38700000e-01]
*****
~~~~

TODO(mengzhou): Implement model loading, calling the ranker with the features, returning the ranking




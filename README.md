# LangRank
by [NeuLab](http://www.cs.cmu.edu/~neulab/) @ [CMU LTI](https://lti.cs.cmu.edu)

LangRank is a program for **Choosing Transfer Languages for Cross-lingual Transfer Learning**, described by our paper on the topic at ACL 2019.
Cross-lingual transfer, where a high-resource *transfer* language is used to improve the accuracy of a low-resource *task* language, is now an invaluable tool for improving performance of natural language processing (NLP) on low-resource languages.
However, given a particular task language, it is not clear *which* language to transfer from, and the standard strategy is to select languages based on *ad hoc* criteria, usually the intuition of the experimenter.
Since a large number of features contribute to the success of cross-lingual transfer (including phylogenetic similarity, typological properties, lexical overlap, or size of available data), even the most enlightened experimenter rarely considers all these factors for the particular task at hand.

LangRank is a program to solve this task of automatically selecting optimal transfer languages, treating it as a ranking problem and building models that consider the aforementioned features to perform this prediction.
In experiments on representative NLP tasks, we have found that LangRank predicts good transfer languages much better than *ad hoc* baselines considering single features in isolation.
Try it out below if you want to figure out which language you should be using to solve your low-resource NLP task!

## Installation

Steps to install:

First install the latest version of lang2vec:

    git clone https://github.com/antonisa/lang2vec.git
    cd lang2vec
    wget http://www.cs.cmu.edu/~aanastas/files/distances.zip .
    mv distances.zip lang2vec/data/
    python3 setup.py install
    
Now clone and install langrank (future: install it as module)

    cd ../
    git clone https://github.com/neulab/langrank.git
    cd langrank
    pip install -r requirements.txt
    wget http://www.cs.cmu.edu/~aanastas/files/indexed.tar.gz .
    tar -xzvf indexed.tar.gz

Further setup (future: provide the pretrained models through wget)

    mkdir -p pretrained/MT
    # copy pretrained .lgbm file there

## Predicting Transfer Languages

You can run check.py to predict transfer languages by providing an unsegmented dataset, a segmented dataset
(using[sentencepiece](https://github.com/google/sentencepiece)) and the language code of your datasets.
    
    python3 check.py -o sample-data/ted-train.orig.aze -s sample-data/ted-train.orig.spm8000.aze -l aze

A detailed walk-through of check.py is provided below. See example in ``check.py`` (it should work if you provide a dataset in your computer).
It follows the example below (ran in the ``langrank`` directory):

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
    >>> prepared = lr.prepare_new_dataset('slv', task="MT", dataset_source=lines)
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


## Citation/Acknowledgements

If you use LangRank, we'd appreciate if you cite the [paper](http://arxiv.org/abs/1903.07926) about it!

    @inproceedings{lin19acl,
        title = {Choosing Transfer Languages for Cross-Lingual Learning},
        author = {Yu-Hsiang Lin and Chian-Yu Chen and Jean Lee and Zirui Li and Yuyan Zhang and Mengzhou Xia and Shruti Rijhwani and Junxian He and Zhisong Zhang and Xuezhe Ma and Antonios Anastasopoulos and Patrick Littell and Graham Neubig},
        booktitle = {The 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
        address = {Florence, Italy},
        month = {July},
        year = {2019}
    }

LangRank was supported by NSF Award #1761548 "Discovering and Demonstrating Linguistic Features for Language Documentation", and the DARPA Low Resource Languages for Emergent Incidents (LORELEI) program under Contract No. HR0011-15-C0114.

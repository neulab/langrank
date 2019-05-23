import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import pytest
from langrank import prepare_train_file, train


def test_train():
    langs = ["aze", "ben", "fin"]
    datasets = [os.path.join(root, "sample-data", "ted-train.orig.{}".format(l)) for l in langs]
    seg_datasets = [os.path.join(root, "sample-data", "ted-train.orig.spm8000.{}".format(l)) for l in langs]
    rank = [[0, 1, 2], [1, 0, 2], [2, 1, 0]] # random
    tmp_dir = "tmp"
    prepare_train_file(datasets=datasets, segmented_datasets=seg_datasets,
                       langs=langs, rank=rank, tmp_dir=tmp_dir, task="MT")
    output_model = "{}/model.txt".format(tmp_dir)
    train(tmp_dir=tmp_dir, output_model=output_model)
    assert os.path.isfile(output_model)

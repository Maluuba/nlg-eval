#!/bin/bash

set -e

function download () {
	URL=$1
	TGTDIR=.
	if [ -n "$2" ]; then
		TGTDIR=$2
		mkdir -p $TGTDIR
	fi
	echo "Downloading ${URL} to ${TGTDIR}"
	wget ${NLG_EVAL_QUIETER_SETUP} $URL -P $TGTDIR
}

function download_file() {
	URL=$1
	filename="${URL##*/}"
	TGTDIR=.
	if [ -n "$2" ]; then
		TGTDIR=$2
	fi
	if [ ! -f "$TGTDIR/$filename" ]; then
		download $@
	fi
}

mkdir --parents nlgeval/data

python -m nltk.downloader punkt

# GloVe word vectors
if python --version 2>&1 | grep -P 'Python 2\.\d'; then
    TEST_PYTHON_VERSION="2"
elif python --version 2>&1 | grep -P 'Python 3\.\d'; then
    TEST_PYTHON_VERSION="3"
fi

if [ "${TEST_PYTHON_VERSION}" == "3" -o -z "${TEST_PYTHON_VERSION+x}" ]; then
    rm -f nlgeval/word2vec/glove2word2vec.py
    download_file https://raw.githubusercontent.com/robmsmt/glove-gensim/dea5e55f449794567f12c79dc12b7f75339b18ba/glove2word2vec.py nlgeval/word2vec
elif [ "${TEST_PYTHON_VERSION}" == "2" ]; then
    rm -f nlgeval/word2vec/glove2word2vec.py
    download_file https://raw.githubusercontent.com/manasRK/glove-gensim/42ce46f00e83d3afa028fb6bf17ed3c90ca65fcc/glove2word2vec.py nlgeval/word2vec
fi

if [ ! -f nlgeval/data/glove.6B.300d.model.bin ]
then
    download http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip glove.6B.300d.txt -d nlgeval/data
    rm -f glove.6B.zip
    PYTHONPATH=`pwd` python nlgeval/word2vec/generate_w2v_files.py
    rm nlgeval/data/glove.6B.300d.txt nlgeval/data/glove.6B.300d.model.txt
fi

# skip-thoughts data
download_file http://www.cs.toronto.edu/~rkiros/models/dictionary.txt nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/utable.npy nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/btable.npy nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl nlgeval/data

# multi-bleu.perl
download_file https://raw.githubusercontent.com/moses-smt/mosesdecoder/b199e654df2a26ea58f234cbb642e89d9c1f269d/scripts/generic/multi-bleu.perl nlgeval/multibleu
[ -e nlgeval/multibleu/multi-bleu.perl ] && chmod +x nlgeval/multibleu/multi-bleu.perl

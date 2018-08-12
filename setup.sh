#!/bin/bash

set -e


if [ ! -f ${NLGEVAL_DATA}/glove.6B.300d.model.bin ]
then
    unzip ${NLGEVAL_DATA}/glove.6B.zip glove.6B.300d.txt -d ${NLGEVAL_DATA}
    PYTHONPATH=`pwd` python nlgeval/word2vec/generate_w2v_files.py
    rm -f ${NLGEVAL_DATA}/glove.6B.zip ${NLGEVAL_DATA}/glove.6B.300d.txt ${NLGEVAL_DATA}/glove.6B.300d.model.txt
fi

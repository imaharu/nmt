#!/bin/sh
correct="../train_data/test.ja"
result=$1
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $result

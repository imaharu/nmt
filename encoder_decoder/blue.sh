#!/bin/sh
correct="../train_data/test.ja"
result="result"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $result

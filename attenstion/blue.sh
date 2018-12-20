#!/bin/sh
correct="../train_data/test.ja"
output="result"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $output

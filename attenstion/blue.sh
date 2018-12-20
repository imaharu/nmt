#!/bin/sh
correct="../train_data/test.ja"
output=$1
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $output

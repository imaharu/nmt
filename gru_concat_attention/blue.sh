#!/bin/sh
correct="../train_data/"
correct=$correct$1
output=$2
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $output

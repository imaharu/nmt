#!/bin/sh
correct="../train_data/$1"
output="trained_model/$2"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $output

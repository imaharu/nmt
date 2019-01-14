#!/bin/sh
correct="../train_data/eval.ja"
output="trained_model/$1"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $output

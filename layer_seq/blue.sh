#!/bin/sh
correct="../train_doc/test.ja"
result="result"
layer="layer.txt"
echo "1 layer"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $result
echo "3 layer"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $layer

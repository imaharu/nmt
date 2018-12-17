#!/bin/sh
correct="test.ja"
result="result"
layer="layer"
echo "1 layer"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $result
echo "3 layer"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $layer

#!/bin/sh
correct="../train_doc/test.ja"
result="result"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $result

#!/bin/sh
correct="test.ja"
output="result"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $output

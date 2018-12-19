#!/bin/sh
correct="../encoder_decoder/test.ja"
output="result"
~/smt/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $output

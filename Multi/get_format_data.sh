#!bin/sh 
tokenizer_command="~/smt/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en"
lowercase_command="~/smt/mosesdecoder/scripts/tokenizer/lowercase.perl"

for file in `find $tac_data -type f | grep english$`; do
	file_name=`basename $file`
	eval $tokenizer_command < $file | eval ${lowercase_command} > "en_data/${file_name}"
done

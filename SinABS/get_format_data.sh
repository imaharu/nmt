#!bin/sh 
tokenizer_command="~/smt/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en"
lowercase_command="~/smt/mosesdecoder/scripts/tokenizer/lowercase.perl"

for file in `find $cnn_data -type f | grep story$`; do
	file_name=`basename $file`
	eval $tokenizer_command < $file | eval ${lowercase_command} > "moses_cnn_data/${file_name}"
done

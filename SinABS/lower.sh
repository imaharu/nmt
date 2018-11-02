#!bin/sh 
lowercase_command="~/smt/mosesdecoder/scripts/tokenizer/lowercase.perl"
for file in `find $basic_cnn -type f | grep story$`; do
    file_name=`basename $file`
    eval $lowercase_command < $file > "moses/${file_name}"
done

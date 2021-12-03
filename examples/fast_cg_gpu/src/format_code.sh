#!/bin/bash
echo "input the file name you want to format"
read file
astyle $file --mode=c --style=bsd --convert-tabs --indent=spaces=4 --indent-continuation=4 --indent-preproc-block --indent-preproc-define --indent-preproc-cond --unpad-paren --align-pointer=name --align-reference=name
rm *.orig

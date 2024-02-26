#!/bin/bash

echo $(which pdflatex)
echo $(which convert)
echo $(pwd)

# Make latex tikz images
for FILE in ./img/*.tex; do
	echo $FILE
	pdflatex -output-directory `dirname $FILE` $FILE > /dev/null 2>&1
	ls img
	convert -density 600 "${FILE%.tex}.pdf" "${FILE%.tex}.png" > /dev/null 2>&1
	mv "${FILE%.tex}.png" ../src/assets/img
        rm "${FILE%.tex}.aux" "${FILE%.tex}.log" "${FILE%.tex}.pdf" 
done

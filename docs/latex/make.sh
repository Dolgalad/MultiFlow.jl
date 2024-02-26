#!/bin/bash

echo "pdflatex path: $(which pdflatex)"
echo "gs path: $(which gs)"
echo "working directory: $(pwd)"

# Make latex tikz images
for FILE in ./img/*.tex; do
	echo $FILE
	pdflatex -output-directory `dirname $FILE` $FILE > /dev/null 2>&1
	PDFFILE="${FILE%.tex}.pdf"
	if [ ! -f "$PDFFILE" ]; then
		echo "PDF output $PDFFILE does not exist. Quitting"
		exit 1
	fi
	PNGFILE="${FILE%.tex}.png"
	gs -dSAFER -r600 -sDEVICE=pngalpha -o "$PNGFILE" "$PDFFILE" > /dev/null 2>&1
	if [ ! -f "$PNGFILE" ]; then
		echo "PNG output $PNGFILE does not exist. Quitting"
		exit 1
	fi
	mv "${FILE%.tex}.png" ../src/assets/img
        rm "${FILE%.tex}.aux" "${FILE%.tex}.log" "${FILE%.tex}.pdf" > /dev/null 2>&1
done

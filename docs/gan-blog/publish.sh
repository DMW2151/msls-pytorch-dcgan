#! /bin/bash

for FILE in ./markdown/*; do 

    htmlout=`(echo $FILE | sed 's/.md$/.html/' | sed 's/markdown/posts/' )`
    
    pandoc --standalone --template ./templates/template.html \
        $FILE -o $htmlout
done
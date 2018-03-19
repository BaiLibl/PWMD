#!/bin/bash

echo 'hello'
i=0
j=0
python p1.py r8test.txt 0
echo 0 >> 1.txt
python test.py label.txt >> 1.txt
for (( i = 1; $i <= 20; i=$i+1 ))
do
    j=$(echo "$j+0.1"|bc)
    echo $j
    python p1.py r8test.txt $j
    echo $j >> 1.txt
    python test.py label.txt >> 1.txt
done

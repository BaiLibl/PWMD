#!/bin/bash

echo 'hello'
i=0
j=0
python p2.py bbcsport.txt 0
echo 0 >> 2.txt
python test2.py label.txt >> 2.txt
for (( i = 1; $i <= 20; i=$i+1 ))
do
    j=$(echo "$j+0.1"|bc)
    echo $j
    python p2.py bbcsport.txt $j
    echo $j >> 2.txt
    python test2.py label.txt >> 2.txt
done

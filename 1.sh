#!/bin/bash

echo 'hello'
i=0
j=0
python p1.py bbcsport.txt 0
echo 0 >> 1.txt
python test.py label.txt >> 1.txt
for (( i = 1; $i <= 100; i=$i+1 ))
do
    j=$(echo "$j+0.01"|bc)
    echo $j
    python p1.py bbcsport.txt $j
    echo $j >> 1.txt
    python test.py label.txt >> 1.txt
done

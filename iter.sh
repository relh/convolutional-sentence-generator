#!/bin/bash
for i in {0..100}
do
  SECONDS=0
  echo $i
  python run.py train
  ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
  echo "Runtime was $ELAPSED"
done

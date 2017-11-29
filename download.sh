#!/bin/sh

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
unzip wiki.en.zip

wget http://norvig.com/big.txt

git pull --recurse-submodules

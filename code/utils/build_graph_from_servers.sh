#!/bin/sh

rm -rf ./whitespace_2/*
rm -rf ./brainfuck_2/*

# scp hamalaa8@lyta.aalto.fi:hacks/charlatan/code/pytorch-a2c-ppo-acktr/tmp/gym/brute/*.csv ./brute/.
scp hamalaa8@lyta.aalto.fi:hacks/charlatan/code/pytorch-a2c-ppo-acktr/tmp/gym/brainfuck_03/*.csv ./brainfuck_2/.
scp hamalaa8@lyta.aalto.fi:hacks/charlatan/code/pytorch-a2c-ppo-acktr/tmp/gym/whitespace-5v/*.csv ./whitespace_2/.
# scp hamalaa8@lyta.aalto.fi:hacks/charlatan/code/pytorch-a2c-ppo-acktr/tmp/gym/brainfuck/*.csv ./
source ~/.bashrc

#python visualize_learning.py --save-name brute_learning --dir ./brute/ --num-mean 100 --num-files 16
python visualize_learning.py --save-name brainfuck_learning --dir ./brainfuck_2/ --num-mean 50 --num-files 16
python visualize_learning.py --save-name whitespace_learning --dir ./whitespace_2/ --num-mean 50 --num-files 16

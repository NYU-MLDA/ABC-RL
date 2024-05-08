#!/bin/bash

RUNID=15001
HOME_DIR=/home/abc586/ABC-RL_public
ROOT_DUMP_DIR=${HOME_DIR}/ABC-RL_ICLR/dump
NUM_RUNS=101

export CUDA_VISIBLE_DEVICES=0
TTSPLIT=5001
REPLAY_MEM=320
BS=320
LOGINFO=test_TTSPLIT${TTSPLIT}_replayMem_${REPLAY_MEM}_BS${BS}_run${RUNID}
nohup python main.py --ttsplit ${TTSPLIT} --library ${HOME_DIR}/lib/7nm/7nm.lib --dumpdir ${ROOT_DUMP_DIR}/${LOGINFO} --runID ${RUNID} --runs ${NUM_RUNS} --replay ${REPLAY_MEM} --bs ${BS} > ${ROOT_DUMP_DIR}/out_${LOGINFO}.txt 2>&1 &


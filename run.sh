#!/bin/sh
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
echo MASTER_PORT=${MASTER_PORT}

export PYTHONPATH=$(pwd):$PYTHONPATH

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

MODEL=$1
MODE=$2
GPUS=$3

if [ -z "$1" ]; then
   GPUS=1
fi

echo $MODEL $MODE $GPUS

# ----------------- kuzushiji-vision -----------
if [[ $MODE == train ]]; then
	echo "==> Training kuzushiji-vision"

	if [[ $MODEL == line ]]; then
		python trainer/train_line_extraction.py

	elif [[ $MODEL == character ]]; then
		if [[ $GPUS -eq 1 ]]; then
            accelerate launch --mixed_precision=bf16 trainer/train_character_detection.py
        else
            accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=$GPUS trainer/train_character_detection.py
        fi
	fi

else
	echo "==> Testing kuzushiji-vision"
	if [[ $MODEL == line ]]; then
		python scripts/visualize_line_extraction.py 

	elif [[ $MODEL == character ]]; then
		python scripts/visualize_character_extraction.py
    fi
fi

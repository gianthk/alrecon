#!/bin/bash
cd /home/beatsbs/PycharmProjects/alrecon

## activate reconstruction environment
eval "$(conda shell.bash hook)"
conda activate tomopy

## launch solara app
solara run alrecon.pages --host localhost
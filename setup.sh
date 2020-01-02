#!/bin/bash

PROJECT_DIR=$(dirname "$0")

printf 'Checking out submodules...'
cd "$PROJECT_DIR" && git submodule update --init --recursive
printf 'done.'

printf '\nSetting up Virtual Environment...'
python3.7 -m venv "$PROJECT_DIR"/_venv
printf 'done.'

printf '\nActivating Virtual Environment...'
. "$PROJECT_DIR/_venv/bin/activate" && printf 'done.' && printf '\nInstalling requirements...' && pip install -qr requirements.txt && printf 'done.'

printf '\nBuilding cython modules...'
cd neural_nets/retina_net && python setup.py build_ext --inplace > /dev/null 2>&1
printf 'done.'

printf '\n\e[32mSetup complete ✓ \e[m\n\n'


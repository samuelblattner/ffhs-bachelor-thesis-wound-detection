#!/bin/bash

PROJECT_DIR=$(dirname "$0")
REQ_FILE=requirements-nogpu.txt

while getopts ":g" opt; do
  case $opt in
    g)
      REQ_FILE=requirements-gpu.txt
      ;;
  esac
done

printf 'Checking out submodules...'
cd "$PROJECT_DIR" && git submodule update --init --recursive > /dev/null 2>&1
printf 'done.'

printf '\nSetting up Virtual Environment...'
python3 -m venv "$PROJECT_DIR"/_venv
printf 'done.'

printf '\nActivating Virtual Environment...'
. "$PROJECT_DIR"/_venv/bin/activate && pip install --upgrade pip > /dev/null 2>&1 && pip install -qr $REQ_FILE
printf 'done.'

printf '\nBuilding cython modules...'
cd neural_nets/retina_net && python setup.py build_ext --inplace > /dev/null 2>&1
cd ..
cd ..
printf 'done.'

printf '\n\e[32mSetup complete ✓ \e[m\n\n'


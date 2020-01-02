# Bachelor Thesis


## Prerequisites
Python >= 3.3
apt-get install python3-venv

## Setup

1. Clone repo
2. Initialize submodules

        $ git submodule update --init --recursive

3. Make & activate Virtual Environment

        $ python -m venv <project-directory>/_venv
        $ . <project-directory>/bin/activate

4. Install requirements

        $ pip install requirements.txt    
   
4. Compile external dependencies

        $ cd neural_nets/retina_net && python setup.py build_ext --inplace
        
5. Add to pythonpath

        $ export PYTHONPATH=$(pwd)/neural_nets/frcnn/:$(pwd)/neural_nets/yolo_3
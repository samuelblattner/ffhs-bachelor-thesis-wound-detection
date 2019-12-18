# Bachelor Thesis


## Setup

1. Clone repo
2. Install virtualenvwrapper
3. Add net submodule directories to virtualenv

        $ add2virtualenv <base dir of repo>/neural_nets/mask_rcnn 
        $ add2virtualenv neural_nets/retina_net/
    
4. Compile external dependencies

        $ cd neural_nets/retina_net && python setup.py build_ext --inplace
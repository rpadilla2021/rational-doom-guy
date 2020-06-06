# rational-doom-guy

## Installation instructions:

### Mac OS:

First, you will have to install all the dependent c library using `brew` whose installation link
is [here](https://brew.sh/). `brew install boost --with-python` will provide with the installation of boost with python integration along with all the co-dependencies for that library itself. You will also need to get: `brew install cmake wget sdl2`. The installations will take time, even if there is no problem.

Next, one needs to install the vizdoom library with `pip install vizdoom`. One will also need to install the python libraries not present in the environment using the command `pip install <library>`.

### Windows:
1. Begin by installing python-3.6.8 onto your computer, specifically install the x86 64-bit launcher of python off of the official python website [here](https://www.python.org/downloads/release/python-368/). We have noticed that some users with Anaconda installed appear to have issues even if they create a new environment with this specific version of python. Thus, it is recommended not to use Anaconda for this project.
1.Clone this repository into whichever directory you would like on your PC.
1.Download the Vizdoom precompiled libraries [here](https://github.com/mwydmuch/ViZDoom/releases/tag/1.1.8pre). After succesfully downloading these precompiled dependencies, be sure to extract and move the folder 'vizdoom' into the 'Lib/site-packages' folder within your 'python36' folder within 'Program Files (x86)' folder.
1.When entering the cloned repo enter 'cd rational-doom-guy'. From here run 'git clone https://github.com/mwydmuch/ViZDoom'.
1. Run 'cd vizdoom/examples/python'
1. If running 'python basic.py' is a success, then you installed everything properly!!

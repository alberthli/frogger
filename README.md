# FRoGGeR: A Fast Robust Grasp Generator

This repository houses the code for the paper "FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric."

## Installation
All code should be run on a Docker container that directly reads and writes from the mounted git directory on your local machine. We use the Drake 1.12.0 release for Ubuntu 22.04 (Jan. 12, 2022). We support Linux machines and theoretically offer MacOS support, though this functionality is currently untested. We do not currently support Windows.

#### Setup on Linux
Install Docker by following the instructions [here](https://docs.docker.com/engine/install/ubuntu/). Make sure to also follow the instructions for [post-installation](https://docs.docker.com/engine/install/linux-postinstall/).

#### Setup on Mac (Untested)
Install Docker Desktop by following the instructions [here](https://docs.docker.com/desktop/install/mac-install/). Also install the latest version of XQuartz [here](https://www.xquartz.org/). In XQuartz, navigate to Preferences > Security and check the "Allow connections from network clients" box.

#### Using MOSEK
If you would like to activate MOSEK as an available commercial solver for optimization programs, then request a free academic license [here](https://www.mosek.com/license/request/) (if you are unsure about which license to choose, select a personal license). Follow the instructions to activate the license, then paste the license file in the root of the repository BEFORE building the Docker image. We recommend doing so if you are going to re-simulate pickups.

#### Starting Docker and Using Multiple Sessions
Clone this repository and `cd` into it. Note that all the data, results, and meshes (including collision meshes) are directly stored on the repository in the `data` directory, with a total size of ~300Mb.

Build the Docker image with a name of your choice by running
```
docker build -t <name_of_image> .
```
This process should take a few minutes, as there are quite a few dependencies (sorry!).

To start a container, run the startup shell script in the root directory of the repo with the below command, which will run terminal commands based on your OS. Any time you want to run the code, you should run this command. The terminal will be running in the container, but all visualization and file changes will be running on your local machine, including plots and simulations in the browser.
```
bash startup.sh -n <name_of_image>
```
To exit the shell in the container, type `exit` into the terminal (or simply close the window).

To create multiple terminal sessions in the container, start one terminal using the startup script and in a separate window type
```
bash startup.sh -j -n <name_of_image>
```
where the `-j` flag denotes "join" to join the existing container.

#### Testing the Installation
In the container, navigate to `~/manipulation` and run the command `python check_installation.py`. This will test for three things:
- It will run a Meshcat simulation of a Kuka iiwa arm. Navigate to `localhost:7000` on your host browser, verify you can move the joints of the robot, and click 'Stop JointSliders'.
- An empty `matplotlib` figure should show up, which confirms the display settings work.
- In the same folder on your host, the file `test.png` should appear. You should manually try to delete the file on the host machine to confirm that read/write permissions are shared between the container and the host. If not, there is an error in the Docker settings. If these permissions are not correct, you can remove the test image by simply rerunning the script again.

## Reproducing and Viewing Results
The main results of the paper can be reproduced by running the script `all_experiments.py` in the `scripts/paper` directory. To view results for individual objects and individual experimental configurations (i.e., our method vs. the baseline), comment out the `run_exp` call in the file. To view only the aggregate results shown in the paper figure, in the container, run in the Python REPL:
```
from core.paper_experiments import summarize_all_results; summarize_all_results()
```

Other scripts used to produce figures or numerical results in the paper can be found in `scripts/paper`, while the script used to clean the YCB meshes is located in `scripts`.

Please direct any questions or concerns to the email of the first author, which can be found in the paper.

<!-- ## Citation
If you found our work useful (either the paper or the code), please use the following citation: INSERT_CITATION_HERE -->

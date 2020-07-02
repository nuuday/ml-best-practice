# AI Python template project
This is a standardized template for python projects in Data and AI.

In the docker image the project folder will be mounted to `/app`, so it is easy to use absolute paths in the code, e.g. the `src/` folder can be found in `/app/src/`.

Some template code can be found in `src/` including example on how to load data without password in plaintext. This template code assumes that the docker setup is used.
Persistent data can be stored i the `data/` folder (only non-sensitive data, aka basically nothing from cubus etc.).

## Starting a new project
1. Do NOT create a repo on BitBucket, this will happen automatically through the setup script.
1. `cd /projects/<USERNAME>`
1. `git clone ssh://git@10.74.139.97:7999/hkrp/template_python_project.git <PROJECTNAME>` - Create a folder for your new project from the template project
1. `cd <PROJECTNAME>`
1. `./setup.sh` - Follow the instructions on the screen. This initializes the project from template and creates the repo in BitBucket.

## Quickguide

#### How to build and run
* Upon starting a new project build a docker image that serves as the base for all work. Run `make build_dev` and follow the instructions on the screen.
* To start working, launch a container using `make start_dev`. If you have a container that is active for this project, it will be stopped and removed before launching a new instance.
* Makefile includes a `make start_prod`, that shows an example of how credentials that are defined inthe Jenkinsfile are parsed to a docker container as environment variables.
#### How to modify images/containers
* If you need to add any libraries or dependencies that are not python, add these to the *Dockerfile_dev* file and rebuild the docker image using `make build_dev`. (E.g if you want to install Vim add the following line `RUN apt-get install -y vim`)
* If you need to add any Python libraries add these to the *requirements.txt* file and rebuild the docker image using `make build_dev`. Notice if you just want to test out a package quickly, simply install these inside the docker container (these packages are lost when the container is closed). You can enter a container by using the command `docker exec -it <CONTAINERNAME> bash` and install a package using `pip install <PACKAGENAME> --user`
#### CPU Constraint
* A CPU core constraint has been added in the target `make start_dev`. The setting is specified with the flag `--cpus=6`, which means 6 cores are used by default. Increase this number if you really need it.
#### Starting a Tensorflow
Running `make start_dev_tf` will start the container with the gpu flag and make a port available for Tensorboard. Dockerfile_dev should be changed to be based on a tensorflow image, e.g. change the first line to `FROM tensorflow/tensorflow:latest-gpu-py3`

#### Useful docker commands
* `docker images` - List all images
* `docker ps -a` - List all containers
* `docker container prune` - Remove stopped containers
* `docker image prune` - Remove old unused versions of images
* `make clean` - prunes containers and images

## How to contribute
If you have any suggestions for changes feel free to create these on remote branch, and submit a pull request to either (phit@tdc.dk or andne@tdc.dk)

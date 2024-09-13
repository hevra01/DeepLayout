#!/bin/bash

#export HOME="/home/hepe00001"
#export PATH="/home/hepe00001/miniconda3/bin:$PATH"

# Define paths
CONDA_ROOT=$HOME/miniconda3
CONDA=${CONDA_ROOT}/bin/conda
#CONDA=/home/hepe00001/miniconda3/bin/conda


# Specify the known environment name
ENV_NAME="layout"

# Check if PROJECT_ROOT is set
if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "'PROJECT_ROOT' is not set. Check that the submit file contains the line 'environment = PROJECT_HOME=\$ENV(PWD)'"
    exit 1
else
    echo "'PROJECT_ROOT=$PROJECT_ROOT'"
fi
echo "home is ${HOME}"
echo "CONDA path is set to: ${CONDA}"

echo "Attempting to run conda from: ${CONDA}"
${CONDA} --version
echo "hevv"
echo "Current PATH: $PATH"


# Check if conda is installed
if [ ! -f ${CONDA} ]; then
  echo "miniconda3 is not installed. Run condor_submit setup.sub first!"
  exit 0
fi

# Check if script argument is provided
if [[ -z "$1" ]]; then
    echo "No Python script specified. Usage: run_task.sh <script_name.py>"
    exit 1
fi

SCRIPT_NAME=$1
echo "$1 $2 $3 $4 $5 $6 $7"

# Execute the script in the conda environment
echo "Running $SCRIPT_NAME in conda env $ENV_NAME"

cd ${PROJECT_ROOT}
${CONDA} run -n ${ENV_NAME} bash -c "
  nvidia-smi
  echo \$CUDA_VISIBLE_DEVICES
  echo \$HOSTNAME
  which python
  python -m pip list
  python $1 $2 $3 $4 $5 $6 $7
"


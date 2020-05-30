#!/bin/bash
if [ "$HOSTNAME" == "gen-uge-submit-p01" ] || [ "$HOSTNAME" == "gen-uge-submit-p02" ]; then
  echo "This script cannot be run from a submit host.  Pleas qlogin and try again."
  exit 1
fi

if hash conda 2>/dev/null; then
  echo "Using conda package manager found at $(command -v conda)"
else
  echo "Using shared conda package manager."
  eval "$(/ihme/covid-19/miniconda/bin/conda shell.bash hook)"
fi

dt=$(date '+%Y-%m-%d_%H-%M-%S') &&
echo "Creating environment covid-deaths-spline-$dt" &&
umask 002
conda create -y --name=covid-deaths-spline-"$dt" -c conda-forge cyipopt gmp python=3.6 &&
conda activate covid-deaths-spline-"$dt" &&
pip install --global-option=build_ext --global-option '-I'$CONDA_PREFIX'/include/' pycddlib &&
pip install drmaa &&
git clone https://github.com/zhengp0/limetr.git &&
cd limetr && make install && cd .. &&
git clone https://github.com/ihmeuw-msca/MRTool.git &&
cd MRTool && git checkout el_hombre_elastico && pip install . && cd .. &&
pip install -e .

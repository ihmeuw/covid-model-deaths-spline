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

echo "Creating environment covid-deaths-spline" &&
umask 002
conda create -y --name=covid-deaths-spline -c conda-forge cyipopt python &&
conda activate covid-deaths-spline &&
git clone https://github.com/zhengp0/limetr.git &&
cd limetr && make install && cd .. &&
git clone https://github.com/ihmeuw-msca/MRTool.git &&
cd MRTool && git checkout seiir_model && python setup.py install && cd .. &&
rm -rf limetr MRTool SLIME &&
python setup.py develop

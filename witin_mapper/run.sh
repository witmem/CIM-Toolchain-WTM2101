##########################################################################
# File Name: run.sh
# Author: afly
# mail: aifei.zhang@witintech.com
# Created Time: Wed Dec  2 17:02:22 2020
#########################################################################
#!/bin/sh

# the pytest will auto find and run the function name start with "test_" in the file
# pytest -s show print() and log message

if [ -d "output" ]; then
  pushd output
    ls | grep -v ".gitkeep" | xargs rm -rf
  popd
fi
rm -rf model/*_*_*_*.onnx
rm -rf model/*_*_*_*.pb
#tensorflow
python3 -m pytest --workers 16 -v tests/tensorflow/test_witin_tensorflow_model.py
EXIT_CODE=$?
if [ $EXIT_CODE != 0 ];then
  echo "witin_mapper test tensorflow failed"
  exit $EXIT_CODE
fi

# onnx
# wtm2100
#python3 -m pytest --workers 16 -v tests/onnx/witin/test_witin_onnx_model_wtm2100.py
#EXIT_CODE=$?
#if [ $EXIT_CODE != 0 ];then
#  echo "witin_mapper test wtm2100 onnx failed"
#  exit $EXIT_CODE
#fi
# wtm2101
python3 -m pytest --workers 16 -v tests/onnx/witin/test_witin_onnx_model_wtm2101.py
EXIT_CODE=$?
if [ $EXIT_CODE != 0 ];then
  echo "witin_mapper test wtm2101 onnx failed"
  exit $EXIT_CODE
fi

# pytorch
python3 -m pytest --workers 16 -v tests/pytorch/witin/test_witin_pytorch_model.py
EXIT_CODE=$?
if [ $EXIT_CODE != 0 ];then
  echo "witin_mapper test pytorch failed"
  exit $EXIT_CODE
fi

##########################################################################
# File Name: install.sh
# Created Time: Tue Oct 27 16:42:58 2020
#########################################################################
#!/bin/sh

echo "-----------------------------------------------------"
echo "-----Welcome to witin witin_mapper install shell-----"
set +x
if [ $? -eq 0 ]; then
    cd "$(dirname "$BASH_SOURCE")"
    CUR_FILE=$(pwd)/$(basename "$BASH_SOURCE")
    CUR_DIR=$(dirname "$CUR_FILE")
    cd - > /dev/null
else
    if [ ${0:0:1} = "/" ]; then
        CUR_FILE=$0
    else
        CUR_FILE=$(pwd)/$0
    fi
    CUR_DIR=$(dirname "$CUR_FILE")
fi

TVM_HOME=${CUR_DIR%/*}

echo "----------------------------------------------------"
grep "TVM_HOME" ~/.bashrc > /dev/null
if [ $? -eq 1 ]; then
    echo "" >> ~/.bashrc
    echo "export TVM_HOME=${TVM_HOME}" >> ~/.bashrc
    echo "export TVM_HOME=${TVM_HOME} >> ~/.bashrc "
	echo "export PYTHONPATH=\$TVM_HOME/python/:\${PYTHONPATH}" >> ~/.bashrc
	echo "export PYTHONPATH=\$TVM_HOME/python/:\${PYTHONPATH} >> ~/.bashrc "
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$TVM_HOME/build/" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$TVM_HOME/build/ >> ~/.bashrc"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$TVM_HOME/build/lib/" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$TVM_HOME/build/lib/ >> ~/.bashrc"
    echo "set TVM_HOME to ~/.bashrc"
    . ~/.bashrc
else
    echo "TVM_HOME is already in ~/.bashrc"
fi

echo "-----------------------------------------------------"

echo "install python module"
pip3 install pillow==5.1.0
pip3 install pytest-parallel


file="/usr/share/fonts/truetype/Gargi/Gargi.ttf"
if [ -f "$file" ]
then
    echo "$file found."
else
    echo "$file not found."
    echo "copy fonts Gargi to /usr/share/fonts/truetype/"
    sudo cp -r ./build/lib/Gargi /usr/share/fonts/truetype/
fi
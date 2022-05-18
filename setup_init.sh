#/bin/bash!

echo "Compile C files"
(cd _receiver/C && make)

echo "[ INFO ] Set file executation rights"
chmod a+x _receiver/C/rtl_daq
chmod a+x _receiver/C/sim
chmod a+x _receiver/C/sync
chmod a+x _receiver/C/gate

sudo chmod +x run.sh
sudo chmod +x kill.sh

echo "[ INFO ] Generate GUI layout"
(cd _GUI/ && exec ./convert.sh)

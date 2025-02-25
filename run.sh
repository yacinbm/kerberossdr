#!/bin/bash
BUFF_SIZE=256 #Must be a power of 2. Normal values are 128, 256. 512 is possible on a fast PC.

# set to /dev/null for no logging, set to some file for logfile. You can also set it to the same file. 
RTLDAQLOG="/dev/null"#"rtl_daq.log"
SYNCLOG="/dev/null"#"sync.log"
GATELOG="gate.log"
PYTHONLOG="python.log"

# If you want to kill all matching processes on startup without prompt. Otherwise, set it to anything else. 
FORCE_KILL="yes"

NPROC=`expr $(nproc) - 1`

### Uncomment the following section to automatically get the IP address from interface wlan0 ###
### Don't forget to comment out "IPADDR="0.0.0.0" ###

# IPADDR=$(ip addr show wlan0 | grep "inet\b" | awk '{print $2}' | cut -d/ -f1)
# while [ "$IPADDR" == "" ] || [ "$IPADDR" == "169.254.*" ]
# do
# sleep 1
# echo "waiting for network"
# IPADDR=$(ip addr show wlan0 | grep "inet\b" | awk '{print $2}' | cut -d/ -f1)
# echo $IPADDR
# done

### End of Section ###

# Useful to set this on low power ARM devices
#sudo cpufreq-set -g performance

# Set for RPI3 with heatsink/fan
#sudo cpufreq-set -d 1.4GHz
# Set for Tinkerboard with heatsink/fan
#sudo cpufreq-set -d 1.8GHz


# Trap SIGINT (2) (ctrl-C) as well as SIGTERM (6), run cleanup if either is caught
trap cleanup 2 6

cleanup() {
	# Kill all processes that have been spawned by this program.
	# we know that these processes have "_receiver", "_GUI" and "_webDisplay" in their names. 
	exec 2> /dev/null           # Suppress "Terminated" message. 
	sudo pkill -f "_receiver" 
	sudo pkill -f "_GUI"
	sudo pkill -f "_webDisplay" 
	
	# also delete all pipes: 
	rm -f _receiver/C/gate_control_fifo
        rm -f _receiver/C/sync_control_fifo
        rm -f _receiver/C/rec_control_fifo
}


# Clear memory
sudo sh -c "echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb"
echo '3' | sudo dd of=/proc/sys/vm/drop_caches status=none

echo "Starting KerberosSDR"

# Check for old processes that could interfere, print warning: 
for string in rtl sim _recei.*sync gate hydra ; do 
    pgrep -af $string 
	if [[ $? -eq 0 ]] ; then 
        if [[ "$FORCE_KILL" != "yes" ]]; then
            read -p "The processes listed above were found and could interfere with the program. Do you want to kill them now? [y|N]" -n1 -r
            echo # newline. 
	    fi
		if [[ "$FORCE_KILL" == "yes" || "$REPLY" =~ ^[Yy]$ ]]
		then
			sudo pkill $string
		else
			echo "OK, not killing these processes. Hope you know what you're doing"
		fi
	fi	
done
		
#sudo kill $(ps aux | grep 'rtl' | awk '{print $2}') 2>$OUTPUT_FD || true


# Enable on the Pi 3 to prevent the internet from hogging the USB bandwidth
#sudo wondershaper wlan0 3000 3000
#sudo wondershaper eth0 3000 3000

sleep 1

# Remake Controller FIFOs. Deleting them should not be neccessary after 
# a clean exit, but why not do it anyway... 
rm -f _receiver/C/gate_control_fifo
mkfifo _receiver/C/gate_control_fifo

rm -f _receiver/C/sync_control_fifo
mkfifo _receiver/C/sync_control_fifo

rm -f _receiver/C/rec_control_fifo
mkfifo _receiver/C/rec_control_fifo

# Start programs at realtime priority levels
curr_user=$(whoami)

# chrt: Set round robin scheduling, with scheduling priority of 50
# taskset: Set the last CPU core for our process
# ionice: Set the IO scheduling class and priority to real-time and most important "0"
# --> Do all of this for the process rtl_daq with arg $BUFF_SIZE
# Redirect stderr (2>) to the file in question

# For GUI, start the gui as most prioritary process and redirect both STDERR and STDOUT to &PYTHONLOG

# The program function as follows:
# The rtl_daq is in charge of capturing data from the RTL_SDR itself using the libusb library. Once it has read
# the data from the SDR, it sends the same data buffer to the next "module" which is sync via the stdout (using 1|).
# You can interface with the rtl_daq module by writing to its FIFO. 
# 

sudo chrt -r 50 taskset -c $NPROC ionice -c 1 -n 0 ./_receiver/C/rtl_daq $BUFF_SIZE 2>$RTLDAQLOG 1| \
	sudo chrt -r 50 taskset -c $NPROC ./_receiver/C/sync $BUFF_SIZE 2>$SYNCLOG 1| \
	sudo chrt -r 50 taskset -c $NPROC ./_receiver/C/gate $BUFF_SIZE 2>$GATELOG 1| \
	sudo nice -n -20 sudo -u $curr_user python3 -O main.py $BUFF_SIZE &>$PYTHONLOG &

echo "KerberosSDR started..."

FMCW Radar implemented on the KerberosSDR

This is a modification of the signal direction detector of the KerberosSDR made to work as a FMCW radar. The system uses a ADF4159 with a triangular FMCW @6GHz with a bandwith of 1.4GHz, swept over 1.4ms. 

The radar is based around 6GHz in order to detect humans in the environment. It works by using two antennas of the KerberosSDR and checking what is the offset in distance detected at both antennas. This distance corresponds to the radius of circles around both antennas, the intersection of which is the position of the target. 

The system can achieve a resolution of ~10cm and is able to distinguish a human body from the surrounding noise, using background substraction.

After setting up the FMCW waveform generator (instructions will come at a later date), launch the program by using the run.sh script. The "Tracker" tab is then used to detect the position of a target in front of the system. 

The system requires the RTL-SDR driver, as described in https://www.rtl-sdr.com/ksdr/, under the "Manually Installing the software on a PC / Other SBC" section.

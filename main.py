from _receiver.hydra_receiver import ReceiverRTLSDR as kerberosSDR
from _signalProcessing.hydra_signal_processor import SignalProcessor
from threading import Thread
import numpy as np
from time import sleep
from sys import argv

def main():
    print("Starting main")
    # Parse arguments
    buffer_size = int(argv[1])
    ip_addr = argv[2]

    receiver = kerberosSDR()
    signal_processor = SignalProcessor(module_receiver=receiver)

    center_freq = 1e9 # Test frequency 1.5GHz
    sample_rate = signal_processor.fs * receiver.decimation_ratio
    gain = [48,48,48,48]

    # Setup receiver
    signal_processor.module_receiver.block_size = buffer_size*1024
    signal_processor.module_receiver.reconfigure_tuner(center_freq, sample_rate, gain)
    sleep(1)

    # synchronize the channels
    signal_processor.module_receiver.switch_noise_source(1) # Turn on noise to sync
    print("starting thread")
    child = Thread(target=signal_processor.run)
    child.start()
    print("synching")
    signal_processor.en_record = True
    signal_processor.en_sync = True # Ask to compute sample offset
    signal_processor.en_sample_offset_sync = True # Ask to send offset to RTLSDR
    signal_processor.en_calib_iq = True # Ask for IQ Correction
    sleep(3) # Wait for signals to stabilize
    print("finished synching")
    signal_processor.en_sync = False # Finish sync process
    signal_processor.module_receiver.switch_noise_source(0) # Turn off noise source

    # Capture data
    signal_processor.en_spectrum = True # Analyze the frequency
    sleep(2)
    for m in range(signal_processor.channel_number):
        fund_freq = np.argmax(signal_processor.spectrum[m+1,:])
        print(f"Channel {m} fundemental frequency at: {fund_freq}MHz")
        print(signal_processor.module_receiver.iq_samples[m,:])

    # Finish
    #signal_processor.stop()
    #signal_processor.module_receiver.close()

if __name__ == "__main__":
    main()

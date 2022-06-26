# binned-volume-matching
Volume Matching algorithm to obtain binned ground radar reflectivity values from space radar observations.
The present version of code runs on Ubuntu 18.04.2.
Extract wradlib-data-master file from wradlib-data-master.tar.xz
To run the volume matching code, follow the instructions in wradlib text file by copy pasting all the commands in the terminal.
Once wradlib is installed, all the environments are set and path to the ground radar and space radar directory file is set one is ready to run the code.
Run: python -i alignment_CAGE_demo.py on the terminal.
The result of the code execution will be two files, one 'matched_reflectivity_data_sweep_2_.txt.mat' and the other 'binned_reflectivity_2_.txt'. 
The 'matched_reflectivity_data_sweep_2_.txt.mat' file contains averaged ground and space radar matched observations whereas 'binned_reflectivity_2_.txt' contains binned ground radar and space radar observations along with the corresponding coordinates. 

April 01, 2024
Version: 1.0

![1](https://github.com/RobinBarta/proPTV/assets/150230392/66660cef-0945-45b3-bc38-355d52a08657)

## proPTV

proPTV is an open source particle tracking velocimetry framework based on the probabilistic approximation of particle tracks. It is written in Python to make 3D PTV more accessible to a wide range of researchers. We hope that freely available software will help the PTV community to grow and connect more frequently. proPTV comes with a numerical test case covering the motion of seeding particles in a DNS of turbulent Rayleigh-Bénard convection in a cubic domain, as seen in the following video. 

![2](https://github.com/RobinBarta/proPTV/assets/150230392/270c145a-d930-4f29-a3f9-fd3fd0d13103)

proPTV is also able to handle cases with high seeding particle densities (>0.1 ppp), as the evaluation with the numerical test case shows. The following figure shows the results of five cases studied, where we illustrate the percentage of matched particles (pmp) by the particle tracks and the percentage of correct tracks (pct). 

![3](https://github.com/RobinBarta/proPTV/assets/150230392/b4ec3cee-b98b-41a7-8d32-dc1aff57c013)

The new probabilistic approximation of particle tracks allows the reconstruction of long and stable particle tracks, which is a key point of the proPTV method. In a test case with 27000 particles and about 0.075 ppp, proPTV was able to generate tracks with a majority as long as the studied case with 30 time steps, as shown in the track length histogram.

![4](https://github.com/RobinBarta/proPTV/assets/150230392/0ac5bd45-88ef-4ab2-9c2f-869a6c8db3c7)

#### Contents

proPTV includes the following tools needed to identify and track the three-dimensional motion of objects in laboratory and field experiments, and to estimate Eulerian velocity and pressure fields from the measured flow.

1) Image Processing to detect particle centres on camera images 
2) Camera calibration tool with the Soloff method
3) PTV tool (Triangulation, Initialization, Prediction and Tracking of particle positions velocities and accelerations)
4) Backtracking and Repairing to refine particle tracks and increase their length
5) Interpolation to Eulerian velocity fields
6) Estimation of incompressible Eulerian velocity and pressure fields
7) Debugging scripts
8) Tools to plot results
9) Post-processing tools

## How to install?

#### Requirements:

proPTV requires Python 3.10 installed with pip. It was developed and tested on Python 3.10.4 which can be downloaded at: https://www.python.org/downloads/release/python-3104/

#### Installation:

1) download the proPTV project to a desired location

2) install python 3.10.4 and pip on Windows or Linux

3) open your terminal, navigate into the proPTV folder location and set up a venv (virtual environment) if you want to use it, else jump to point 5.

  `python -m venv venv`

4) active the venv one Windows with 

  `cd venv/Scripts activate`
   
   and on Linux with:

  `source venv/bin/activate`

5) install the required python packages

  `pip install -r requirements.txt`

6) Optional: I recommand installing spyder to execute proPTV.

  `pip install spyder`

  You can open spyder with typing

  `spyder`

  in the terminal.
  
## How to use proPTV?

proPTV contains two folders, the code folder stores the executable Python scripts and the data folder stores the input and output files of each processed experiment. All you need to provide are raw camera images of the seeded particles named c{cam}_{time}.tif and images of a calibration plate at different depth positions to the view axis of each camera. Note that the number of cameras, i.e. cam, ranges from 0 to NumberOfCameras-1. When you have the required input files, navigate to code/preProcessing and follow the steps:

1) load the script in the 1_createDataset folder, modify the parameter class header, and run the script to create your dataset in the data/ folder. Each dataset contains three folders, analysis, which stores the postprocessed images, input, which stores all input files, the calibration parameters, and the lists of determined image particle center positions

2) copy the raw camera images and the calibration target images to the folders in data/YourDataset/input.

3) rename your files with 2_renameFiles and convert them into TIF images using 3_convertImagesToTIF to the required name: c{cam}_{time}.tif

4) run script 4_createMasks to create masks of the raw camera images for each camera

5) run script 5_imageProcessing to detect the particle centres on the camera images.

6) run the 6_getMarker script to identify the markers of your calibration target for each camera and store them per plane in the xyXYZ format, which holds the image coordinate xy for each 3D marker position XYZ. You can select the desired coordinate system for each camera by changing the parameter class in the header. Run the combineMarker.py script inside the folder to combine each xyXYZ list for each plane into the final markerList.txt for each camera, which will be used in the next step.

7) run 7_calibration to estimate the calibration parameter based on the markerLists.txt for each camera. Run 8_testCalibration to check your calibration result. You can refine the calibration parameters with 9_VSC if needed.

8) go to code/main and modify the config.py file with the parameters to process your dataset

9) run code/main/proPTV.py, which will create tracks and save them to data/YourDataset/output/tracks

10) to run the backtracking method, change the backtracking option in the config file to True and load the saved tracks after using proPTV.py for the first time.

11) you can perform the following post-processing routines located in code/postProcessing: repair, smoothing, coordinate transformation of the tracks, interpolation to Eulerian velocity fields, estimation of a divergence-free Eulerian velocity field and assimilation of pressure fields
   
12) using the scripts in code/postProcessing/8_getResults your can plot all tracks or track length histograms.

Please have a look at the HowToUse.mp4 tutorial where I explain how to use proPTV on our numerical test case of turbulent Rayleigh-Bénard convection. The numerical test case can be generated with the data/makeData/makeData.py script for a specified number of particles.

## How to cite?

When proPTV is useful for your scientific work, you may cite us as:

[1] Barta, Robin, et al. "proPTV - A probabilistic particle tracking velocimetry framework" Journal of Computational Physics (2023) (under review)

[2] Herzog, Sebastian, et al. "A probabilistic particle tracking framework for guided and Brownian motion systems with high particle densities." SN computer science 2.6 (2021): 485, https://doi.org/10.1007/s42979-021-00879-z

and include the licence file in all copies with modifications or other code that uses parts of the proPTV framework.

## Contact

If you have a question or need help installing proPTV or fixing a bug you have found, please contact me: robin.barta@dlr.de

I am happy to help and look forward to meeting you.

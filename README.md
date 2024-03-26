April 01, 2024
Version: 1.0

![label](https://github.com/RobinBarta/proPTV/assets/150230392/b84320c0-18cd-4b46-8f1a-2bad9ecdd143)

## proPTV

proPTV is an open source particle tracking velocimetry framework based on the probabilistic approximation of particle tracks. It is written in Python to make 3D PTV more accessible to a wide range of researchers. We hope that freely available software will help the PTV community to grow and connect more frequently. proPTV comes with a numerical test case covering the motion of seeding particles in a DNS of turbulent Rayleigh-Bénard convection in a cubic domain, as seen in the following video. 

![movie](https://github.com/RobinBarta/proPTV/assets/150230392/68dcf636-17f5-4d10-9449-67cc67fd3103)

proPTV is also able to handle cases with high seeding particle densities (>0.1 ppp), as the evaluation with the numerical test case shows. The following figure shows the results of five cases studied, where we illustrate the percentage of matched particles (pmp) by the particle tracks and the percentage of correct tracks (pct). 

![EvalTrack](https://github.com/RobinBarta/proPTV/assets/150230392/df5f941e-2fc9-4864-9105-e26d4c624f55)

The new probabilistic approximation of particle tracks allows the reconstruction of long and stable particle tracks, which is a key point of the proPTV method. In a test case with 27000 particles and about 0.075 ppp, proPTV was able to generate tracks with a majority as long as the studied case with 30 time steps, as shown in the track length histogram.

![allTracks_histogram](https://github.com/RobinBarta/proPTV/assets/150230392/fea05946-66ac-4c6d-8b62-3f4aa4773d7c)

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

proPTV requires Python 3.10 installed with pip. It was developed and tested on Python 3.10.4 which can be downloaded at:
  
  `https://www.python.org/downloads/release/python-3104/`

#### Installation:

1) download the proPTV project to a desired location

2) install python 3.10.4 and pip on Windows or Linux

3) open your terminal, navigate into the proPTV folder and set up a venv (virtual environment)

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

Please have a look at the HowToUse.mp4 tutorial where I explain how to use it on our numerical test case of turbulent Rayleigh-Bénard convection.

## How to cite?

When proPTV is useful for your scientific work, you may cite us as:

[1] Barta, Robin, et al. "proPTV - A probabilistic particle tracking velocimetry framework" Journal of Computational Physics (2023) (under review)

[2] Herzog, Sebastian, et al. "A probabilistic particle tracking framework for guided and Brownian motion systems with high particle densities." SN computer science 2.6 (2021): 485, https://doi.org/10.1007/s42979-021-00879-z

and include the licence file in all copies with modifications or other code that uses parts of the proPTV framework.

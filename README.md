# **Automated Conformer Sampling and Refinement using ORCA**

This repository contains a streamlined Python code for conformer sampling and refinement using **ORCA 6.0**. The code automates the process of progressively refining the level of theory, eliminating the need for manual intervention. This code is meant for HPC slurm submission system, minor modifications must be made if you use HPC outside of our group (UTD). Using an input yaml file we are able to automate the process of submitting calculations and then choosing a sampling method to choose the favored conformations, to then refine the calculation with more precise methods.

---

## **Features**
- Automated workflow for conformer sampling.
- Gradual refinement of the computational level in subsequent steps.
- Reduces manual errors and ensures efficient resource utilization.

---

## **Requirements**
1. **Python 3.x**: The code is written in Python and requires the following libraries:
   -numpy 
   -yaml 
   -pandas
2. **QORCA**: This is a Python script in our Group Repo for Orca submission
---

## **Required Files**
The code requires **ORCA input files** and ***Input yaml file*** to function:

1. **First Input File**:  
   - This file is the initial ORCA file and must include **GOAT specifications** for conformer optimization at the CHEAP level of theory i.e XTB.  

2. **Initial XYZ** (`step1.xyz`):  
   - This file should be used for the intial geometry.
   - Example line for coordinates:  
    
3. **N Input File** (`stepN.inp`):  
   - Depening on how many steps of refinement are in your calculation you need to provide their input files.  
   - It is **recommended** to include a frequency calculation for accurate results at the final step.
   - Input files must have the final line have molecule.xyz.
   ```
     * xyz 0 1 molecule.xyz
   ```
An example folder with input files (`Examples/`) is provided for reference.

---
## **Example Usage**
- `python auto_goat.py -c 128 input.yaml`
  
If you want to run the code in the background and not disconnect when you log off HPC:
- `nohup python auto_goat.py input.yaml -c 128 &`



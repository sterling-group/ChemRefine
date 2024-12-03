# **Automated Conformer Sampling and Refinement using ORCA**

This repository contains a streamlined Python code for conformer sampling and refinement using **ORCA 6.0**. The code automates the process of progressively refining the level of theory, eliminating the need for manual intervention.

---

## **Features**
- Automated workflow for conformer sampling.
- Gradual refinement of the computational level in subsequent steps.
- Reduces manual errors and ensures efficient resource utilization.

---

## **Requirements**
1. **ORCA 6.0**: Ensure ORCA is installed and added to your system's `PATH`.
2. **Python 3.x**: The code is written in Python and requires the following libraries:
   - `os`
   - `subprocess`
   - `numpy`
3. **QORCA**: This is a Python script in our Group Repo for Orca submission
---

## **Required Files**
The code requires **three ORCA input files** to function:

1. **First Input File**:  
   - This file is the initial ORCA file and must include **GOAT specifications** for conformer optimization at the CHEAP level of theory i.e XTB.  

2. **Second Input File** (`step2.inp`):  
   - This file should use a higher level of theory for subsequent refinement.
   - Example line for coordinates:  
     ```
     * xyz 0 1 molecule.xyz
     ```
3. **Third Input File** (`step3.inp`):  
   - The highest level of theory is used in this file.  
   - It is **recommended** to include a frequency calculation for accurate results.

An example folder with input files (`Examples/`) is provided for reference.

---



#!/usr/bin/env python3

"""
ChemRefine: Automated conformer sampling and refinement using ORCA.

A streamlined Python package for conformer sampling and refinement using ORCA 6.0.
Automates the process of progressively refining the level of theory, eliminating 
the need for manual intervention in HPC SLURM environments.

Copyright (C) 2025 Sterling Research Group

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__version__ = "0.3.0"
__author__ = "Sterling Research Group"
__email__ = "dal950773@utdallas.edu"

from .core import ChemRefiner

__all__ = ["ChemRefiner", "run_mlff_calculation", "MLFFJobSubmitter"]

#!/usr/bin/env python3
"""
Backward compatibility wrapper for auto_goat.py

This script maintains backward compatibility for users who expect to run:
python auto_goat.py input.yaml

It simply imports and calls the main function from the new package structure.

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

import sys
import os

# Add src to path to import autogoat package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from autogoat.auto_goat import main

if __name__ == "__main__":
    main()

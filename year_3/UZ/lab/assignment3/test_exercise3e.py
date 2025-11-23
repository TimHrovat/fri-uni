#!/usr/bin/env python3

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import os

# Add current directory to path
sys.path.append('.')

# Import the functions we need
from main import exercise3e

if __name__ == "__main__":
    print("Testing Exercise 3E implementation...")
    try:
        exercise3e()
        print("Exercise 3E completed successfully!")
    except Exception as e:
        print(f"Error running exercise3e: {e}")
        import traceback
        traceback.print_exc()
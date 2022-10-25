#!/usr/bin/env python3

import argparse
from importlib.resources import path
import numpy as np  
import cv2
import face_recognition
import os
import time
from copy import deepcopy
from detection import Detector
from Recognizer import Recognizer
import keyboard

def main():
    try:
        while True:
            Detector()
            Recognizer()
           
    except KeyboardInterrupt:
        pass
         
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
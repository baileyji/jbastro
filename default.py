#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits

def parse_cl():
    parser = argparse.ArgumentParser(description='Help undefined',
                                     add_help=True)
    parser.add_argument('-p','--port', dest='PORT',
                        action='store', required=False, type=int,
                        help='')
    parser.add_argument('--log', dest='LOG_LEVEL',
                        action='store', required=False, default='',
                        type=str,
                        help='')
    return parser.parse_args()

if __name__ =='__main__':
    args=parse_cl()

import argparse
import re
import ast
import matplotlib.pyplot as plt

from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bmark_output_paths',
        type=str,
        nargs='+',
        help='paths to bmark output files'
    )
    parser.add_argument(
        '--nvsmi_output_paths',
        type=str,
        nargs='+',
        help='paths to nvsmi output files'
    )
    args = parser.parse_args()
    main(args)


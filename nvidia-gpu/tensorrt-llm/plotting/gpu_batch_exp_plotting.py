import argparse
import re

from pathlib import Path


def main(args):
    bmark_output_paths = args.bmark_output_paths.split()
    nvsmi_output_paths = args.nvsmi_output_paths.split()
    assert(len(bmark_output_paths) == len(nvsmi_output_paths))

    for i in range(len(bmark_output_paths)):
        bmark_output_path = bmark_output_paths[i]
        nvsmi_output_path = nvsmi_output_paths[i]
        print(f'PATHS: {bmark_output_path} {nvsmi_output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bmark_output_paths',
        type=str,
        required=True,
        help='paths to bmark output files'
    )
    parser.add_argument(
        '--nvsmi_output_paths',
        type=str,
        required=True,
        help='paths to nvsmi output files'
    )
    args = parser.parse_args()
    main(args)

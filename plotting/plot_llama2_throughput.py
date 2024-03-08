import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for llama2 throughput experiments.')
    parser.add_argument(
        '--input-file-path',
        required=True,
        type=str,
        help='The path to the JSON output file.'
    )

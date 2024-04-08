import subprocess
import argparse
import re


power_usage_pattern = r"\d+W / \d+W"


def get_nvsmi_info_V100S_PCIE_32GB():
    output = subprocess.check_output(['nvidia-smi'])
    decoded_output = output.decode('utf-8')
    for line in decoded_output.split('\n'):
        #print(line)
        power_usage_match = re.search(power_usage_pattern, line)

        if power_usage_match:
            print("Found: ", power_usage_match.group())
    #print(output)


def main(args):
    # create output dir and file
    #with (args.output_dir / args.output_file).open('w') as f:
    #    f.write(f'engine path: {args.engine_dir}\n')
    #    f.write(f'tokenizer path: {args.tokenizer_dir}\n')
    get_nvsmi_info_V100S_PCIE_32GB()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='directory for saving output files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='output file name'
    )
    args = parser.parse_args()
    main(args)

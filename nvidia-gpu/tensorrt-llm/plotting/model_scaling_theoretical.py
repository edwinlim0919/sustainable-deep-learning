import argparse
import matplotlib.pyplot as plt


def calculate_model_flops(
    model_name,
    sequence_length
):



def main(args):
    if args.generate_plot == 'theoretical_dev_1':
        # compare calflops empirical results from calculations based on "Scaling Laws for Neural Language Models"
        sequence_lengths = [256]
        model_names = ['meta-llama/Llama-2-7b-chat-hf']
        calculate_model_flops(
            sequence_lengths,
            model_names
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_plot',
        type=str,
        required=True,
        help='specify the name of the plot to generate'
    )
    args = parser.parse_args()
    main(args)

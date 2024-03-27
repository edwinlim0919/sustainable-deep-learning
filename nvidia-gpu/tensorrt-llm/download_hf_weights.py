import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='utility script for downloading weights/checkpoints from HugginFace')
    parser.add_argument(
        '--model-name',
        required=True,
        type=str,
        help='the name of the model on HuggingFace'
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.save_pretrained(f'./{args.model_name}_model')
    tokenizer.save_pretrained(f'./{args.model_name}_tokenizer')

#!/usr/bin/env bash
# Script to convert Llama2 weights to run with Hugging Face
#
# Arguments
#   args[0] : llama2 source weight folder name
#   args[1] : llama2 destination weight folder name
#   args[2] : model size
#
# Example Usage
#   ./convert_llama2_weights.sh llama-2-7b-chat llama-2-7b-chat-hf 7B
#   ./convert_llama2_weights.sh llama-2-13b-chat llama-2-13b-chat-hf 13B


if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_weight_folder> <model_size> <destination_weight_folder>"
    exit 1
fi


src_weights=$1
dest_weights=$2
model_size=$3
target_file="./${src_weights}/tokenizer.model"
echo "source weights: $src_weights"
echo "destination weights: $dest_weights"
echo "model size: $model_size"
echo "target file: $target_file"


if [ -f $target_file ]; then
    echo "Target file $target_file already exists. Removing it..."
	rm "$target_file"
fi


ln ./tokenizer.model "$target_file"
TRANSFORM=`python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')"`
python $TRANSFORM --input_dir ./${src_weights} --model_size ${model_size} --output_dir ./${dest_weights}

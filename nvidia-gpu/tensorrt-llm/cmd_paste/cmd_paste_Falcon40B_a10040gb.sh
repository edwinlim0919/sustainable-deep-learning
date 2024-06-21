# TensorRT-LLM standalone

sed -i "s/92a3527e8c38/5e4d136c9647/g" cmd_paste_Falcon40B_a10040gb.sh

# /TensorRT-LLM/examples/falcon
pip install -r requirements.txt
apt-get update
apt-get -y install git git-lfs
git lfs install
git clone https://huggingface.co/tiiuae/falcon-40b-instruct falcon/40b-instruct


# /TensorRT-LLM/examples/falcon
# 4-way tensor parallelism + 1-way pipeline parallelism
python3 convert_checkpoint.py --model_dir ./falcon/40b-instruct --dtype bfloat16 --output_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --tp_size 4 --load_by_shard

# max_batch_size 1
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1/ --workers 4 --max_batch_size 1

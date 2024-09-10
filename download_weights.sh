apt update
apt install vim screen -y 
pip uninstall torch -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu124 # full options are cpu/cu118/cu121/cu124
pip install -e .["dev"]
tune download Skywork/Skywork-Reward-Llama-3.1-8B --output-dir /workspace/reward/ --ignore-patterns " "
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /workspace/policy_model --ignore-patterns "original/consolidated.00.pth" --hf-token $HF_TOKEN

tune download TinyLlama/TinyLlama_v1.1 --output-dir /workspace/policy_model_1b
tune download smohammadi/tinyllama_rm_sentiment_1b  --output-dir /workspace/reward_1b --ignore-patterns " "

mkdir /workspace/ppo
mkdir /workspace/ppo/policy/
mkdir /workspace/ppo/value/

mkdir /workspace/ppo_1b
mkdir /workspace/ppo_1b/policy/
mkdir /workspace/ppo_1b/value/

wandb login
echo "tune run lora_ppo_finetune_single_device --config recipes/configs/llama3_1/8b_ppo_qlora.yaml"
pip3 install torch torchao --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install -e .["dev"]
tune download Skywork/Skywork-Reward-Llama-3.1-8B --output-dir /tmp/reward/ --ignore-patterns " "
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/policy_model --ignore-patterns "original/consolidated.00.pth"
mkdir /tmp/ppo
mkdir /tmp/ppo/policy/
mkdir /tmp/ppo/value/

echo "tune run lora_ppo_finetune_single_device --config recipes/configs/llama3_1/8b_ppo_qlora.yaml"
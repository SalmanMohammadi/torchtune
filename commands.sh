tune run ppo_full_finetune_single_device --config mistral/7B_full_ppo_low_memory \
metric_logger=torchtune.training.metric_logging.WandBLogger \
metric_logger.project=ppo_v2 \
metric_logger.name=baseline_main_0 \
num_steps=65536 \
tokenizer.path=/workspace/Mistral-7B-Instruct-v0.2/tokenizer.model \
checkpointer.checkpoint_dir=/workspace/Mistral-7B-Instruct-v0.2/ \
ref_policy_checkpointer.checkpoint_dir=/workspace/Mistral-7B-Instruct-v0.2/ \
value_checkpointer.checkpoint_dir=/workspace/RM-Mistral-7B/ \
reward_checkpointer.checkpoint_dir=/workspace/RM-Mistral-7B/ \
log_peak_memory_stats=True \
tokenizer.max_seq_len=1024 \
compile=False 


-

tune run ppo_full_finetune_single_device --config mistral/7B_full_ppo_low_memory \
metric_logger=torchtune.training.metric_logging.WandBLogger \
metric_logger.project=ppo_v2 \
metric_logger.name=ppo_v2 \
num_steps=65536 \
tokenizer.path=/workspace/Mistral-7B-Instruct-v0.2/tokenizer.model \
checkpointer.checkpoint_dir=/workspace/Mistral-7B-Instruct-v0.2/ \
ref_policy_checkpointer.checkpoint_dir=/workspace/Mistral-7B-Instruct-v0.2/ \
value_checkpointer.checkpoint_dir=/workspace/RM-Mistral-7B/ \
reward_checkpointer.checkpoint_dir=/workspace/RM-Mistral-7B/ \
log_peak_memory_stats=True \
tokenizer.max_seq_len=1024 \
compile=False 

-

tune run ppo_full_finetune_single_device --config mistral/7B_full_ppo_low_memory \
metric_logger=torchtune.training.metric_logging.WandBLogger \
metric_logger.project=ppo_v2 \
metric_logger.name=ppo_v2 \
num_steps=65536 \
tokenizer.path=/workspace/Mistral-7B-Instruct-v0.2/tokenizer.model \
checkpointer.checkpoint_dir=/workspace/Mistral-7B-Instruct-v0.2/ \
ref_policy_checkpointer.checkpoint_dir=/workspace/Mistral-7B-Instruct-v0.2/ \
value_checkpointer.checkpoint_dir=/workspace/RM-Mistral-7B/ \
reward_checkpointer.checkpoint_dir=/workspace/RM-Mistral-7B/ \
log_peak_memory_stats=True \
tokenizer.max_seq_len=1024 \
compile=True 
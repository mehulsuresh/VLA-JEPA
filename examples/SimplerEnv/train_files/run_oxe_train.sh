export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

export WANDB_MODE=disabled


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/SimplerEnv/train_files/vlajepa_ft.yaml \
  #--framework.name ${Framework_name} \
  #-datasets.vla_data.data_root_dir ${oxe_data_root}\
  #--datasets.vla_data.data_mix ${data_mix} \
  #--datasets.vla_data.per_device_batch_size 16 \
  #--trainer.freeze_modules ${freeze_module_list} \
  #--trainer.max_train_steps 100000 \
  #--trainer.save_interval 10000 \
  #--trainer.logging_frequency 100 \
  #--trainer.eval_interval 1000 \
  #--run_root_dir ${run_root_dir} \
  #--run_id ${run_id} \
  #--wandb_project starVLA_simplerEnv \
  #--wandb_entity jinhuiye \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####

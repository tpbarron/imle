# TODO: set LOG_DIR for your machine!
# LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/10_20_17/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/10_20_17/";
LOG_DIR="test_dir";

BASELINES_MUJOCO_ARGS="--bnn-n-updates-per-step 500 --max-episode-steps 1000 --num-processes 1 --num-steps 2048 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 5000000 --use-gae"
BASELINES_ATARI_ARGS="--bnn-n-updates-per-step 500 --max-episode-steps 1000 --num-processes 4 --num-steps 256 --entropy-coef 0 --ppo-epoch 4 --lr 1e-3 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 5000000 --use-gae"

declare -a args_types=("mujoco" "atari")
declare -a cam_types=("follow" "fixed")

for seed in 1 2 3 4 5
do
  for args_type in "${args_types[@]}"
  do
    for cam_type in "${cam_types[@]}"
    do
      for framestack in 3 4
      do
        for bnn_update_interval in 1 5
        do
          if [ args_type == 'mujoco' ]
          then
            args=BASELINES_MUJOCO_ARGS
          elif [ args_type == 'atari' ]
          then
            args=BASELINES_ATARI_ARGS
          fi
          EXP_PATH="${LOG_DIR}/baseline/hopper_vision_dense_rew_x_t1000_args${args_type}_cam${cam_type}_framestack${framestack}/${i}/";
          echo $EXP_PATH
          mkdir -p $EXP_PATH
          python main.py ${args} --seed ${seed} --log-dir ${EXP_PATH} --env-name HopperVisionBulletX-v0 --cam-type ${cam_type} --num-stack ${framestack} --bnn-update-interval ${bnn_update_interval}

          EXP_PATH="${LOG_DIR}/imle_baseline/hopper_vision_dense_rew_x_t1000_enc_norm_args${args_type}_cam${cam_type}_framestack${framestack}_bnnupint${bnn_update_interval}/${i}/";
          echo $EXP_PATH
          mkdir -p $EXP_PATH
          python main.py ${args} --seed ${seed} --imle --log-dir ${EXP_PATH} --env-name HopperVisionBulletX-v0 --cam-type ${cam_type} --num-stack ${framestack}  --bnn-update-interval ${bnn_update_interval}

          EXP_PATH="${LOG_DIR}/imle_baseline/hopper_vision_dense_rew_x_t1000_enc_norm_eta_decay_args${args_type}_cam${cam_type}_framestack${framestack}_bnnupint${bnn_update_interval}/${i}/";
          echo $EXP_PATH
          mkdir -p $EXP_PATH
          python main.py ${args} --seed ${seed} --imle --log-dir ${EXP_PATH} --env-name HopperVisionBulletX-v0 --cam-type ${cam_type} --num-stack ${framestack}  --bnn-update-interval ${bnn_update_interval} --eta-decay
        done
      done
    done
  done
done

# # alternate imle / no explr
# for i in 1
# do
#   EXP_PATH="${LOG_DIR}/baseline/hopper_vision_dense_rew_x_t1000/${i}/";
#   mkdir -p $EXP_PATH
#   python main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --env-name HopperVisionBulletX-v0
#
#   EXP_PATH="${LOG_DIR}/imle_baseline/hopper_vision_dense_rew_x_t1000_enc_norm/${i}/";
#   mkdir -p $EXP_PATH
#   python main.py ${DEFAULT_ARGS} --imle --seed ${i} --log-dir ${EXP_PATH} --env-name HopperVisionBulletX-v0
#
#   # tmux new-session -s imle_hopper_norm_$i -d "python main.py ${DEFAULT_ARGS} --imle --seed ${i} --log-dir ${EXP_PATH} --env-name HopperVisionBulletX-v0"
#
#   # EXP_PATH="${LOG_DIR}/imle_baseline/hopper_dense_rew_x_t1000_enc_norm_eta_decay/${i}/";
#   # mkdir -p $EXP_PATH
#   # tmux new-session -s imle_hopper_norm_eta_decay_$i "python main.py ${DEFAULT_ARGS} --imle --seed ${i} --log-dir ${EXP_PATH} --env-name HopperVisionBulletX-v0 --eta-decay"
#
#   # while tmux has-session
#   # do
#   #   sleep 1;
#   # done
#
#   # tmux new-session -s ppo_hopper_$i -d "python main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --env-name HopperVisionBulletX-v0"
#
#   # # All this to run two tasks at a time. Probably a better way but I'm not a bash expert
#   # while tmux has-session
#   # do
#   #   sleep 1;
#   # done
# done

# LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/10_17_17/";
LOG_DIR="/media/Backup/trevor1_data/data/outputs/imle/10_20_17/set2/";
# LOG_DIR="test_dir/";
DEFAULT_ARGS="--bnn-n-updates-per-step 500 --max-episode-steps 1000 --num-processes 1 --num-steps 2048 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 1000000 --use-gae"

# no explr
# low dim
for i in 1 2 3 4 5
do
  EXP_PATH="${LOG_DIR}/imle_baseline/half_cheetah_dense_rew_x_t1000_enc_norm/${i}/";
  mkdir -p $EXP_PATH
  tmux new-session -s imle_half_cheetah_norm_$i -d "python main.py ${DEFAULT_ARGS} --imle --seed ${i} --log-dir ${EXP_PATH} --env-name HalfCheetahBulletX-v0"

  EXP_PATH="${LOG_DIR}/imle_baseline/half_cheetah_dense_rew_x_t1000_enc_norm_eta_decay/${i}/";
  mkdir -p $EXP_PATH
  tmux new-session -s imle_half_cheetah_norm_eta_decay_$i "python main.py ${DEFAULT_ARGS} --imle --seed ${i} --log-dir ${EXP_PATH} --env-name HalfCheetahBulletX-v0 --eta-decay"

  # All this to run two tasks at a time. Probably a better way but I'm not a bash expert
  while true
  do
    running=0;
    if tmux has-session -t imle_half_cheetah_norm_$i;
    then
      running=1;
      sleep 1;
    fi
    if tmux has-session -t imle_half_cheetah_norm_eta_decay_$i;
    then
      running=1;
      sleep 1;
    fi
    if [ $running -eq 0 ];
    then
      break;
    fi
  done

  EXP_PATH="${LOG_DIR}/baseline/half_cheetah_dense_rew_x_t1000/${i}/";
  mkdir -p $EXP_PATH
  tmux new-session -s ppo_half_cheetah_$i -d "python main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --env-name HalfCheetahBulletX-v0"

  EXP_PATH="${LOG_DIR}/vime_baseline/half_cheetah_dense_rew_x_t1000/${i}/";
  mkdir -p $EXP_PATH
  tmux new-session -s vime_half_cheetah_$i -d "python main.py ${DEFAULT_ARGS} --vime --seed ${i} --log-dir ${EXP_PATH} --env-name HalfCheetahBulletX-v0"

  # All this to run two tasks at a time. Probably a better way but I'm not a bash expert
  while true
  do
    running=0;
    if tmux has-session -t ppo_half_cheetah_$i;
    then
      running=1;
      sleep 1;
    fi
    if tmux has-session -t vime_half_cheetah_$i;
    then
      running=1;
      sleep 1;
    fi
    if [ $running -eq 0 ];
    then
      break;
    fi
  done

done

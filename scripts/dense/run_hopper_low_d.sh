# LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/10_17_17/";
LOG_DIR="/media/Backup/trevor1_data/data/outputs/imle/10_20_17/";
# LOG_DIR="test_dir/";
DEFAULT_ARGS="--bnn-n-updates-per-step 500 --max-episode-steps 1000 --num-processes 1 --num-steps 2048 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 1000000 --use-gae"

# no explr
# low dim
for i in 1 2 3 4 5
do
  EXP_PATH="${LOG_DIR}/imle_baseline/hopper_dense_rew_x_t1000_enc_norm/${i}/";
  mkdir -p $EXP_PATH
  tmux new-session -s imle_hopper_norm_$i -d "python main.py ${DEFAULT_ARGS} --imle --seed ${i} --log-dir ${EXP_PATH} --env-name HopperBulletX-v0"

  EXP_PATH="${LOG_DIR}/imle_baseline/hopper_dense_rew_x_t1000_enc_norm_eta_decay/${i}/";
  mkdir -p $EXP_PATH
  tmux new-session -s imle_hopper_norm_eta_decay_$i "python main.py ${DEFAULT_ARGS} --imle --seed ${i} --log-dir ${EXP_PATH} --env-name HopperBulletX-v0 --eta-decay"

  while tmux has-session
  do
    sleep 1;
  done

  EXP_PATH="${LOG_DIR}/baseline/hopper_dense_rew_x_t1000/${i}/";
  mkdir -p $EXP_PATH
  tmux new-session -s ppo_hopper_$i -d "python main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --env-name HopperBulletX-v0"

  EXP_PATH="${LOG_DIR}/vime_baseline/hopper_dense_rew_x_t1000/${i}/";
  mkdir -p $EXP_PATH
  tmux new-session -s vime_hopper_$i -d "python main.py ${DEFAULT_ARGS} --vime --seed ${i} --log-dir ${EXP_PATH} --env-name HopperBulletX-v0"

  # All this to run two tasks at a time. Probably a better way but I'm not a bash expert
  while tmux has-session
  do
    sleep 1;
  done

done


# # imle
# for i in 1
# do
#   EXP_PATH="${LOG_DIR}/imle_baseline/hopper_dense_rew/${i}/";
#   mkdir -p $EXP_PATH
#   python main.py $DEFAULT_ARGS --imle --seed $i --log-dir $EXP_PATH --env-name "HopperBulletX-v0"
# done

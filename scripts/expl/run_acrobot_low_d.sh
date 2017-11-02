LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/nips_deeprl/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/10_18_17/";
DEFAULT_ARGS="--bnn-n-updates-per-step 500 --max-episode-steps 1000 --num-processes 1 --num-steps 2048 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 250000 --use-gae"

# imle comparison
for i in 1 2 3
do
  # ENV_NAME="${LOG_DIR}/baseline/acrobot_continuous_x_t1000/${i}/";
  # mkdir -p $ENV_NAME;
  # python main.py $DEFAULT_ARGS --seed $i --log-dir $ENV_NAME --env-name "AcrobotContinuousX-v0" &

  ENV_NAME="${LOG_DIR}/imle_baseline/acrobot_continuous_x_linbnn_t500/${i}/";
  mkdir -p $ENV_NAME;
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir $ENV_NAME --env-name "AcrobotContinuousX-v0" &

  # ENV_NAME="${LOG_DIR}/vime_baseline/acrobot_continuous_x_t1000/${i}/";
  # mkdir -p $ENV_NAME;
  # python main.py $DEFAULT_ARGS --vime --seed $i --log-dir $ENV_NAME --env-name "AcrobotContinuousX-v0" &

  # wait;
done

# # vime baselines
# for i in 1 2 3 4 5
# do
#   mkdir -p "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/"
#   python main.py $DEFAULT_ARGS --vime --seed $i --log-dir "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/out.log"
# done
#
# # no explr
# # low dim
# for i in 1 2 3 4 5
# do
#   mkdir -p "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/"
#   python main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/out.log"
# done

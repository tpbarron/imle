
LOG_DIR="/home/trevor/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--num-processes 8 --num-steps 20 --batch-size 160"

# vime baselines
for i in 1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/"
  python main.py $DEFAULT_ARGS --vime --seed $i --log-dir "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/out.log"
done

# imle comparison
for i in 1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/imle_baseline/acrobot_continuous_x/${i}/"
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}/imle_baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}/imle_baseline/acrobot_continuous_x/${i}/out.log"
done

# no explr
# low dim
for i in 1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/"
  python main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/out.log"
done

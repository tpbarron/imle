LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/";
# LOG_DIR="/media/Backup/trevor1_data/data/outputs/imle/";
DEFAULT_ARGS="--num-processes 8 --num-steps 20 --batch-size 160"

# imle vision
for i in 1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/imle_baseline/mountaincar_continuous_vision_x/${i}/"
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}/imle_baseline/mountaincar_continuous_vision_x/${i}/" --env-name "MountainCarContinuousVisionX-v0" --num-stack 4 | tee  "${LOG_DIR}/imle_baseline/mountaincar_continuous_vision_x/${i}/out.log"
done

# no explr
# high dim
for i in 1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/baseline/mountaincar_continuous_vision_x/${i}/"
  python main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}/baseline/mountaincar_continuous_vision_x/${i}/" --env-name "MountainCarContinuousVisionX-v0" --num-stack 4 | tee  "${LOG_DIR}/baseline/mountaincar_continuous_vision_x/${i}/out.log"
done

#! /bin/bash
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu=4GB
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1

LOG_DIR="/data/obs006/imle/";
DEFAULT_ARGS="--num-processes 16 --num-steps 20 --batch-size 160"
# DEFAULT_ARGS="--num-processes 8 --num-steps 20 --batch-size 160"
export LOCALDIR="/home/obs006/src/IMLE/imle"
i=${run}
j=${example}

echo "Changing to ${LOCALDIR}: Example ${j} Run: ${i}"
cd ${LOCALDIR}

case $j in
1)
	mkdir -p "${LOG_DIR}vime_baseline/mountaincar_continuous_x/${i}/"
	python3 main.py $DEFAULT_ARGS --vime --seed $i --log-dir "${LOG_DIR}vime_baseline/mountaincar_continuous_x/${i}/" --env-name "MountainCarContinuousX-v0" | tee  "${LOG_DIR}vime_baseline/mountaincar_continuous_x/${i}/out.log"
	;;
2)
	mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_x/${i}/"
	python3 main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}imle_baseline/acrobot_continuous_x/${i}/out.log"
	;;
3)
	mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_x/${i}/"
	python3 main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_x/${i}/" --env-name "MountainCarContinuousX-v0" | tee  "${LOG_DIR}imle_baseline/mountaincar_continuous_x/${i}/out.log"
	;;
4)
	mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/${i}/"
	python3 main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/${i}/" --env-name "AcrobotContinuousVisionX-v0" --num-stack 4 | tee  "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/${i}/out.log"
	;;
5)
	mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/${i}/"
	python3 main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/${i}/" --env-name "MountainCarContinuousVisionX-v0" --num-stack 4 | tee  "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/${i}/out.log"
	;;
6)
	mkdir -p "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/"
	python3 main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}vime_baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/out.log"
	;;
7)
	mkdir -p "${LOG_DIR}/baseline/mountaincar_continuous_x/${i}/"
	python3 main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}vime_baseline/mountaincar_continuous_x/${i}/" --env-name "MountainCarContinuousX-v0" | tee  "${LOG_DIR}/baseline/mountaincar_continuous_x/${i}/out.log"
	;;
8)
	mkdir -p "${LOG_DIR}/baseline/acrobot_continuous_vision_x/${i}/"
	python3 main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/${i}/" --env-name "AcrobotContinuousVisionX-v0" --num-stack 4 | tee  "${LOG_DIR}/baseline/acrobot_continuous_vision_x/${i}/out.log"
	;;
9)
	mkdir -p "${LOG_DIR}/baseline/mountaincar_continuous_vision_x/${i}/"
	python3 main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/${i}/" --env-name "MountainCarContinuousVisionX-v0" --num-stack 4 | tee  "${LOG_DIR}/baseline/mountaincar_continuous_vision_x/${i}/out.log"
	;;
esac


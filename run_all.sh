#!/usr/bin/env bash
BASE_CFG=config.yaml
DS_CONFIG=config/deepspeed_config.json
DS_CMD="deepspeed --num_gpus 4 main.py"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

KERNELS=(
  rbf 
)

try_batch_sizes=(256 128 64 32)

# 배치 사이즈 변경 함수
update_batch_size() {
  local bs=$1

  yq eval -i ".TRAIN.BATCH_SIZE = $bs" $BASE_CFG
  yq eval -i ".VAL.BATCH_SIZE = $bs" $BASE_CFG

  sed -i -E "s/(\"train_micro_batch_size_per_gpu\"\s*:\s*)[0-9]+/\1$bs/" $DS_CONFIG
}

# 배치 사이즈 기억용 전역 변수 (associative array)
declare -A batch_size_cache

# deepspeed 실행 및 CUDA OOM 에러 감지 시 배치 낮추며 재시도
# $1: run_name_base
# $2: (optional) 이미 성공한 batch size 있어 시작할 배치 사이즈
run_deepspeed_with_batch_retry() {
  local run_name_base=$1
  local start_bs=${2:-}

  local sizes=()
  if [ -n "$start_bs" ]; then
    sizes+=("$start_bs")
    for bs in "${try_batch_sizes[@]}"; do
      if (( bs < start_bs )); then
        sizes+=("$bs")
      fi
    done
  else
    sizes=("${try_batch_sizes[@]}")
  fi

  for bs in "${sizes[@]}"; do
    update_batch_size $bs
    yq eval -i ".WANDB.NAME = \"transformer_mlp_weather_200_epochs_${run_name_base}_bs${bs}\"" $BASE_CFG

    echo "▶▶ 실행 시도: ${run_name_base}_bs${bs} (batch size=$bs)"

    err_log=$(mktemp)
    $DS_CMD --cfg "$BASE_CFG" 2> $err_log
    RET=$?

    if [ $RET -eq 0 ]; then
      rm -f $err_log
      echo "성공: ${run_name_base}_bs${bs}"
      batch_size_cache["$run_name_base"]=$bs
      return 0
    else
      if grep -qi "out of memory" $err_log || grep -qi "cuda runtime error" $err_log || grep -qi "torch\.OutOfMemoryError" $err_log; then
        echo "!!! CUDA Out of Memory 감지, batch size 줄여서 재시도 ..."
        rm -f $err_log
      else
        echo "기타 오류 발생. 로그 확인 필요."
        cat $err_log
        rm -f $err_log
        return $RET
      fi
    fi
  done

  echo "모든 배치 사이즈로 시도했으나 실패: ${run_name_base}"
  return 1
}

# 고정 배치 사이즈로 단순 실행 (batch size 조정 없이 실행 실패 시 종료)
run_deepspeed_fixed_batch() {
  local run_name_base=$1
  local bs=$2

  update_batch_size $bs
  yq eval -i ".WANDB.NAME = \"transformer_mlp_weather_200_epochs_${run_name_base}_bs${bs}\"" $BASE_CFG
  echo "▶▶ 실행 (batch size 고정): ${run_name_base}_bs${bs}"

  err_log=$(mktemp)
  $DS_CMD --cfg "$BASE_CFG" 2> $err_log
  RET=$?

  if [ $RET -eq 0 ]; then
    rm -f $err_log
    echo "성공: ${run_name_base}_bs${bs}"
    return 0
  else
    echo "실패: ${run_name_base}_bs${bs}, 로그 확인 필요"
    cat $err_log
    rm -f $err_log
    return $RET
  fi
}

# 실험 루프 공통
run_experiment() {
  local run_name_base=$1
  local kernel=$2
  local group_size=$3
  local stride=$4

  # 실험에 필요한 공통 파라미터 세팅
  yq eval -i ".MODEL.LOSS_NAMES = [\"MSE_MMD_Loss\"]" $BASE_CFG
  yq eval -i ".MODEL.LOSS_USE_MSE = true" $BASE_CFG   # 항상 MSE만 사용
  yq eval -i ".MMD.KERNEL = \"$kernel\"" $BASE_CFG
  yq eval -i ".MMD.GROUP_SIZE = $group_size" $BASE_CFG
  yq eval -i ".MMD.STRIDE = $stride" $BASE_CFG

  # 배치사이즈 찾기/적용 및 실험
  if [ -n "${batch_size_cache[$run_name_base]}" ]; then
    local cached_bs=${batch_size_cache[$run_name_base]}
    echo "정보 있음: ${run_name_base} 이미 성공한 배치 사이즈 ${cached_bs} 있음. 그 배치부터 시도"
    run_deepspeed_with_batch_retry "$run_name_base" "$cached_bs"
  else
    run_deepspeed_with_batch_retry "$run_name_base"
  fi
}

# (1) group_size == stride
for K in "${KERNELS[@]}"; do
  for G in 16 8 4 2; do
    S=$G
    RUN_NAME="MMD_${K}_gs${G}_st${S}"
    run_experiment "$RUN_NAME" "$K" "$G" "$S"
  done
done

# (2) group_size = 2,4,8,16 / stride = 절반
for K in "${KERNELS[@]}"; do
  for G in 16 8 4 2; do
    S=$((G/2))
    RUN_NAME="MMD_${K}_gs${G}_st${S}"
    run_experiment "$RUN_NAME" "$K" "$G" "$S"
  done
done

# (3-1) group_size == stride for group sets
GROUP_SETS=("8,16" "4,16" "2,4" "2,4,8" "2,4,8,16")
for GROUP_SET in "${GROUP_SETS[@]}"; do
  IFS=',' read -ra GRPS <<< "$GROUP_SET"
  for K in "${KERNELS[@]}"; do
    for G in "${GRPS[@]}"; do
      S=$G
      RUN_NAME="MMD_${K}_gs${GROUP_SET}_st_equal"
      run_experiment "$RUN_NAME" "$K" "$G" "$S"
    done
  done
done

# (3-2) group_size = 그룹 / stride = 그룹 절반
for GROUP_SET in "${GROUP_SETS[@]}"; do
  IFS=',' read -ra GRPS <<< "$GROUP_SET"
  for K in "${KERNELS[@]}"; do
    for G in "${GRPS[@]}"; do
      S=$((G / 2))
      RUN_NAME="MMD_${K}_gs${GROUP_SET}_st_half"
      run_experiment "$RUN_NAME" "$K" "$G" "$S"
    done
  done
done
#!/usr/bin/env bash
set -eo pipefail

ROOT=`dirname "$0"`
ROOT=`cd "$ROOT/.."; pwd`
export OUTPUT=$ROOT/output
export OUTPUT_BIN=$ROOT/build
export DATA_ROOT=$ROOT/Data
export CASE_ROOT=$ROOT/bin
export LOG_ROOT=$ROOT/log
export gpu_type=`nvidia-smi -q | grep "Product Name" | head -n 1 | awk '{print $NF}'`
source $ROOT/bin/run_clas_mkl_benchmark.sh
# test model type
model_type="static"
if [ $# -ge 1 ]; then
    model_type=$1
fi
export MODEL_TYPE=${model_type}

# test run-time device
device_type="gpu"
if [ $# -ge 2 ]; then
    device_type=$2
fi

mkdir -p $LOG_ROOT

echo "==== run ${MODEL_TYPE} model benchmark ===="

if [ "${MODEL_TYPE}" == "static" ]; then
    if [ "${device_type}" == "gpu" ]; then
        bash $CASE_ROOT/run_clas_gpu_trt_benchmark.sh "${DATA_ROOT}/PaddleClas/infer_static"
        bash $CASE_ROOT/run_det_gpu_trt_benchmark.sh "${DATA_ROOT}/PaddleDetection/infer_static"
        bash $CASE_ROOT/run_clas_int8_benchmark.sh "${DATA_ROOT}/PaddleClas/infer_static"
        bash $CASE_ROOT/run_det_int8_benchmark.sh "${DATA_ROOT}/PaddleDetection/infer_static"
    elif [ "${device_type}" == "cpu" ]; then 
        export KMP_AFFINITY=granularity=fine,compact,1,0
        export KMP_BLOCKTIME=1
        # no_turbo 1 means turning off turbo, it was set to save power. no_turbo 0 means turning on turbo which will improve some performance
        # echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
        echo "==== CPU ===="
        default_cpu_batch_size=(1 2 4)
        cpu_batch_size=${3:-${default_cpu_batch_size[@]}}
        default_cpu_num_threads=(1 2 4)
        cpu_num_threads=${4:-${default_cpu_num_threads[@]}}
        run_clas_mkl_func "${DATA_ROOT}/PaddleClas/infer_static" cpu_batch_size cpu_num_threads
        # bash $CASE_ROOT/run_det_mkl_benchmark.sh "${DATA_ROOT}/PaddleDetection/infer_static"  # very slow
    fi
elif [ "${MODEL_TYPE}" == "dy2static" ]; then
    bash $CASE_ROOT/run_clas_gpu_trt_benchmark.sh "${DATA_ROOT}/PaddleClas/infer_dygraph"
    # bash $CASE_ROOT/run_dy2staic_det_gpu_trt_benchmark.sh
fi

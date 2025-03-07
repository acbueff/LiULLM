#! /bin/bash

#SBATCH --job-name=nemo-curator:example-script
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --account=trustllm
#SBATCH --partition=devel

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =================================================================
# Begin easy customization
# =================================================================

# module restore curator

# Base directory for all SLURM job logs and files
# Does not affect directories referenced in your script
# Make sure to mount the directories your script references
export BASE_DIR=`pwd`
export BASE_JOB_DIR=$BASE_DIR/nemo-curator-jobs
export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID
export VENV=$BASE_DIR/.env

# Logging information
export LOGDIR=$JOB_DIR/logs
export PROFILESDIR=$JOB_DIR/profiles
export SCHEDULER_FILE=$LOGDIR/scheduler.json
export SCHEDULER_LOG=$LOGDIR/scheduler.log
export DONE_MARKER=$LOGDIR/done.txt

# Main script to run
# In the script, Dask must connect to a cluster through the Dask scheduler
# We recommend passing the path to a Dask scheduler's file in a
# nemo_curator.utils.distributed_utils.get_client call like the examples
    #--input-data /p/scratch/trustllm/WP2/FHG/web/OSCAR/sv/2014-42_sv_meta/*.jsonl 
export DEVICE='cpu'
export SCRIPT_PATH=$BASE_DIR/add_id.py
export SCRIPT_COMMAND="python $SCRIPT_PATH \
    --scheduler-file $SCHEDULER_FILE \
    --device $DEVICE \
    --text-field text \
    --id-field id \
    --hash-field hash \
    --partition 10GiB \
    --input-data $BASE_DIR/2024/*/*.json.gz \
    --output-data $BASE_DIR/test_out/*.json.gz \
    --add-id cellar_{ix} \
    --add-hash \
    "

# Below must be path to entrypoint script on your system
# export CONTAINER_ENTRYPOINT=$BASE_DIR/slurm/container-entrypoint.sh
export ENTRYPOINT=$BASE_DIR/container-entrypoint.sh
# Container parameters
# export CONTAINER_IMAGE=$BASE_DIR/nemo_24.03.framework.sif

# Network interface specific to the cluster being used
export INTERFACE=ib0
export PROTOCOL=tcp

# CPU related variables
# 0 means no memory limit
export CPU_WORKER_MEMORY_LIMIT=0

# GPU related variables
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"
export RMM_SCHEDULER_POOL_SIZE="1GB"
export RMM_WORKER_POOL_SIZE="72GiB"
export LIBCUDF_CUFILE_POLICY=OFF


# =================================================================
# End easy customization
# =================================================================

mkdir -p $LOGDIR
mkdir -p $PROFILESDIR

srun $ENTRYPOINT

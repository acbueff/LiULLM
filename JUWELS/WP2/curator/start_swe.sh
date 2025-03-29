#! /bin/bash

#SBATCH --job-name=dedup
#SBATCH --account=trustllm-eu
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --partition=mem192

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
# ENVIRONMENT
# =================================================================

#module restore dedup
#export PYTHON=python
module load Python/3.11.3
source /p/project1/trustllm-eu/cubagyllensten1/curator/env2/bin/activate
#export ENV=/p/project1/trustllm-eu/cubagyllensten1/curator/env2
#export PYTHON=$ENV/bin/python
#export VENV=/p/project1/trustllm-eu/cubagyllensten1/curator/venv/bin/activate
#export PYTHON=/p/project/trustllm-eu/cubagyllensten1/curator/venv/bin/python

# =================================================================
# Begin easy customization
# =================================================================

# Base directory for all SLURM job logs and files
# Does not affect directories referenced in your script
export BASE_JOB_DIR=`pwd`/logs
export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID

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
export DEVICE='cpu'
export LANG=swe
export SCRIPT_PATH=dedup.py
export SCRIPT_COMMAND="python $SCRIPT_PATH $LANG $SCHEDULER_FILE"

# Network interface specific to the cluster being used
export INTERFACE=ib0
export PROTOCOL=tcp

# CPU related variables
# 0 means no memory limit
export CPU_WORKER_MEMORY_LIMIT=0

# =================================================================
# End easy customization
# =================================================================

mkdir -p $LOGDIR
mkdir -p $PROFILESDIR

# Start the container
srun wrap.sh

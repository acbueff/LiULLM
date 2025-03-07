#! /bin/bash

#SBATCH --job-name=quality_signals
#SBATCH --account=trustllm-eu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=0-8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
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
# ENVIRONMENT
# =================================================================

module load Python/3.11.3
source /p/project1/trustllm-eu/cubagyllensten1/curator/env2/bin/activate

echo ${SLURM_CPUS_PER_TASK}

#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

srun python quality_signals.py /p/data1/trustllmd/WP2/data/final /p/data1/trustllmd/WP2/data/signals $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT

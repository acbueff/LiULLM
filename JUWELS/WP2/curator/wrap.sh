#! /bin/bash

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

#source $ENV/bin/activate #/p/project1/trustllm-eu/cubagyllensten1/curator/venv/bin/activate
source /p/project1/trustllm-eu/cubagyllensten1/curator/env2/bin/activate

# Start the scheduler on the rank 0 node
if [[ -z "$SLURM_NODEID" ]] || [[ $SLURM_NODEID == 0 ]]; then
  echo "Starting scheduler"
  dask scheduler \
  --scheduler-file $SCHEDULER_FILE \
  --protocol $PROTOCOL \
  --interface $INTERFACE >> $SCHEDULER_LOG 2>&1 &
fi

# Wait for the scheduler to start
sleep 60

# Start the workers on each node
echo "Starting workers..."
export WORKER_LOG=$LOGDIR/worker_${SLURM_NODEID}-${SLURM_LOCALID}.log
dask worker \
--scheduler-file $SCHEDULER_FILE \
--memory-limit $CPU_WORKER_MEMORY_LIMIT \
--local-directory $SCRATCH_trustllm_eu/cubagyllensten1 \
--nworkers $NWORKERS \
--preload setup_worker.py \
--interface $INTERFACE >> $WORKER_LOG 2>&1 &

# Wait for the workers to start
sleep 120

if [[ -z "$SLURM_NODEID" ]] || [[ $SLURM_NODEID == 0 ]]; then
  echo "Starting $SCRIPT_PATH"
  bash -c "$SCRIPT_COMMAND"
  echo "FINISHED"
  touch $DONE_MARKER
fi

# All nodes wait until done to keep the workers and scheduler active
while [ ! -f $DONE_MARKER ]
do
  sleep 15
done

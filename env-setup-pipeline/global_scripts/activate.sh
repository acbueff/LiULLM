#!/usr/bin/env bash

# Configure the software environment used outside the container.

set -euo pipefail

# Do not use these variables; they may be overwritten.
# Instead, use `get_curr_file` or `get_curr_dir` after sourcing `get_curr_file.sh`.
_curr_file="${BASH_SOURCE[0]:-${(%):-%x}}"
_curr_dir="$(dirname "$_curr_file")"
source "$_curr_dir"/get_curr_file.sh "$_curr_file"

# Load global configuration.
source "$(get_curr_dir)"/global_configuration.sh

# Machine-specific activation steps.
if [ "$machine_name" = jsc ] \
       || [ "$machine_name" = jwb ] \
       || [ "$machine_name" = jwc ] \
       || [ "$machine_name" = jrc ]; then
    source "$(get_curr_dir)"/../jsc/_activate.sh
elif [ "$machine_name" = bsc ] \
       || [ "$machine_name" = mn5a ]; then
    source "$(get_curr_dir)"/../bsc/_activate.sh
elif [ "$machine_name" = berzelius ]; then
    # New branch for Berzelius HPC.
    source "$(get_curr_dir)"/../berz/_activate.sh
else
    echo "Warning: No machine-specific activation script found for machine '$machine_name'."
fi

# Load configuration specific to the LLM Foundry environment.
source "$(get_curr_dir)"/configuration.sh

# Configure HuggingFace libraries cache.
source "$(get_curr_dir)"/../env_cache_scripts/configure_caches.sh

pop_curr_file

#!/usr/bin/env bash
# Partially configure the software environment used outside the container for different HPC systems.

set -euo pipefail

# Do not use these variables directly; instead, use get_curr_file and get_curr_dir.
_curr_file="${BASH_SOURCE[0]:-${(%):-%x}}"
_curr_dir="$(dirname "$_curr_file")"
# Adjust the relative path if your directory is named "global_scripts" (underscore) instead of "global-scripts"
source "$_curr_dir"/../global_scripts/get_curr_file.sh "$_curr_file"

# Load global configuration (which should set machine_name and container_library).
source "$(get_curr_dir)"/../global_scripts/global_configuration.sh

case "$machine_name" in
    juwelsbooster | juwels | jurecadc)
        module purge
        module load Stages/2025
        module load GCC
        if [ "$container_library" = "apptainer" ]; then
            module try-load Apptainer-Tools
            if ! command -v apptainer >/dev/null 2>&1; then
                echo "Could not find Apptainer on JSC machine."
                exit 1
            fi
        fi
        ;;
    berzelius)
        module purge
        # For Berzelius, we replace "SystemModule/2025" with modules that exist.
        # For example, we load a build environment module and a GCC module available on Berzelius.
        module load buildenv-nvhpc/24.5-cuda12.4
        module load buildenv-gcccuda/12.1.1-gcc12.3.0
        if [ "$container_library" = "apptainer" ]; then
            module try-load Apptainer-Tools
            if ! command -v apptainer >/dev/null 2>&1; then
                echo "Could not find Apptainer on Berzelius."
                exit 1
            fi
        fi
        ;;
    *)
        echo "No specific configuration for machine: $machine_name"
        ;;
esac

pop_curr_file

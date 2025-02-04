#!/usr/bin/env bash
# Partially configure the software environment used outside the container for different HPC systems.

set -euo pipefail

# Ensure SYSTEMNAME is defined. If not, default it to "berzelius".
: "${SYSTEMNAME:=berzelius}"

echo "Activating environment for SYSTEMNAME: ${SYSTEMNAME}"

# Do not use these variables directly; instead, use get_curr_file and get_curr_dir.
_curr_file="${BASH_SOURCE[0]:-${(%):-%x}}"
_curr_dir="$(dirname "$_curr_file")"
# Adjust the relative path if your directory is named "global_scripts" (underscore) instead of "global-scripts"
source "$_curr_dir"/../global_scripts/get_curr_file.sh "$_curr_file"

source "$(get_curr_dir)"/../global_scripts/global_configuration.sh

case "$SYSTEMNAME" in
    juwelsbooster | juwels | jurecadc)
        module purge
        module load Stages/2025
        module load GCC
        if [ "$container_library" = apptainer ]; then
            module try-load Apptainer-Tools
            if ! command -v apptainer >/dev/null 2>&1; then
                echo "Could not find Apptainer on JSC machine."
                exit 1
            fi
        fi
        ;;
    berzelius)
        module purge
        # Replace the following module loads with those required on the Berzelius HPC.
        module load SystemModule/2025
        module load GCC/12.1.0
        if [ "$container_library" = apptainer ]; then
            module try-load Apptainer-Tools
            if ! command -v apptainer >/dev/null 2>&1; then
                echo "Could not find Apptainer on Berzelius."
                exit 1
            fi
        fi
        ;;
    *)
        echo "No specific configuration for SYSTEMNAME: $SYSTEMNAME"
        ;;
esac

pop_curr_file

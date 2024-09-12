#!/bin/bash
# Function to print usage
usage() {
    echo "Usage: $0 --project <project_id> [--tpu_name <tpu_name>] [--zone <zone>] [--verbose]"
    echo "The --project flag is mandatory."
    echo "If --zone is not specified, us-central2-b will be used by default."
    exit 1
}
# Parse command line arguments
ZONE="us-central2-b"
VERBOSE=false
PROJECT=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tpu_name) PROVIDED_TPU_NAME="$2"; shift ;;
        --zone) ZONE="$2"; shift ;;
        --project) PROJECT="$2"; shift ;;
        --verbose) VERBOSE=true ;;
        *) usage ;;
    esac
    shift
done
# Check if project is provided
if [ -z "$PROJECT" ]; then
    echo "Error: --project flag is mandatory"
    usage
fi
# Function to find an available TPU
find_tpu() {
    local found_tpu=$(gcloud compute tpus list --project $PROJECT --zone=$ZONE --format="value(name)" --limit=1 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "Error: Failed to list TPUs. Please check your gcloud configuration and permissions."
        exit 1
    fi
    echo $found_tpu
}
# Set TPU_NAME
if [ -z "$PROVIDED_TPU_NAME" ]; then
    TPU_NAME=$(find_tpu)
    if [ -z "$TPU_NAME" ]; then
        echo "No TPU found in zone $ZONE. Please make sure you have an active TPU or specify a TPU name."
        exit 1
    fi
    echo "No TPU name provided. Using automatically found TPU: $TPU_NAME"
else
    TPU_NAME=$PROVIDED_TPU_NAME
    echo "Using provided TPU name: $TPU_NAME"
fi
# Verify TPU existence
if ! gcloud compute tpus describe $TPU_NAME --project $PROJECT --zone=$ZONE &>/dev/null; then
    echo "Error: TPU '$TPU_NAME' not found in zone $ZONE. Please check the TPU name and zone."
    exit 1
fi
# Print selected TPU, zone, and project
echo "Using TPU '$TPU_NAME' in zone '$ZONE' for project '$PROJECT'"
# Function to run command on all TPU VM workers
run_on_all_workers() {
    local command="$1"
    local output=$(gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE --worker=all --command "$command" 2>&1)
    if [ $? -ne 0 ]; then
        echo "Error: Failed to execute command on TPU workers. Please check your TPU status and permissions."
        exit 1
    fi
    echo "$output"
}
# Kill processes using /dev/accel0
echo "Checking for processes using /dev/accel0..."
kill_output=$(run_on_all_workers "sudo lsof -t /dev/accel0 | xargs -r sudo kill -9")
if [ -n "$kill_output" ]; then
    echo "Processes killed on TPU workers:"
    echo "$kill_output"
else
    echo "No processes found using /dev/accel0"
fi
# Print system information if verbose flag is set
if [ "$VERBOSE" = true ]; then
    echo "Printing system information..."
    system_info=$(run_on_all_workers "uname -a && lscpu")
    echo "$system_info"
fi
echo "All operations completed."
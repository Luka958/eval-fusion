#!/bin/bash

set -e

install_dependencies() {
    local dir=$1
    local is_root=$2

    cd "$dir" || { echo "Error: Failed to enter directory $dir"; exit 1; }

    if [ ! -f "pyproject.toml" ]; then
        echo "Error: Directory $dir does not contain pyproject.toml!"
        exit 1
    fi

    if [ "$is_root" = true ]; then
        echo "Installing dependencies in root..."
        poetry install --no-root
    else
        echo "Installing dependencies in $dir..."
        poetry install
    fi
}

ROOT_DIR="$(dirname "$(realpath "$0")")"

SUB_DIRS=(
    "libs/core"
    "libs/community/deepeval"
    "libs/community/mlflow"
    "libs/community/phoenix"
    "libs/community/ragas"
    "libs/community/trulens"
    "libs/vendor/openai"
    "libs/vendor/vertexai"
    "libs/test"
)

install_dependencies "$ROOT_DIR" true

for sub_dir in "${SUB_DIRS[@]}"; do
    dir="$ROOT_DIR/$sub_dir"

    if [ ! -d "$dir" ]; then
        echo "Error: Directory $dir does not exist!"
        exit 1
    fi

    install_dependencies "$dir" false
done

cd "$ROOT_DIR"

echo "Poetry dependencies installation completed."

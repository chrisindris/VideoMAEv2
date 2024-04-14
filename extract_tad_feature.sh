#!/bin/bash
# shell script wrapper for extract_tad_feature.py

for ARGUMENT in "$@"; do
        KEY=$(echo $ARGUMENT | cut -f1 -d=)

        KEY_LENGTH=${#KEY}
        VALUE="${ARGUMENT:$KEY_LENGTH+1}"

        export "$KEY"="$VALUE"
done

# --- Flags and Default Values ---



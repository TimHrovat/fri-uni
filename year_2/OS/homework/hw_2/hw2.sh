#!/bin/bash

if [ $# -lt 2 ]; then
    echo "This program takes at least two arguments."
    exit 1
fi

base_arg="$1"

for ((i = 2; i <= $#; i++)); do
    cur_arg="${!i}"

    if [ "$base_arg" = "$cur_arg" ]; then
        echo "Argument $i: '$cur_arg' is equal to first argument '$base_arg'"
    else
        echo "Argument $i: '$cur_arg' is not equal to first argument '$base_arg'"
    fi
done

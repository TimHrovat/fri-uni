#!/bin/bash

count=0

for arg in "$@"; do
    transformed=$(echo "$arg" | sed 's/a/ha/g; s/e/he/g; s/i/hi/g; s/o/ho/g; s/u/hu/g')

    echo "$count: $transformed"

    ((count++))
done

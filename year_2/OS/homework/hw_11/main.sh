#!/bin/bash

N=${1:-100}
counter_file="stevec.txt"
echo "0" >"$counter_file"
echo "Začetna vrednost: 0"

total_procs=$((2 * N))

for ((i = 0; i < total_procs; i++)); do
    if [ $i -lt "$N" ]; then
        type="inc"
    else
        type="dec"
    fi

    (
        if [ "$2" = "sync" ]; then
            flock -x "$counter_file" sh -c '
                value=$(cat '"$counter_file"')
                echo "$1 $2: prebrano $value"
                if [ "$2" = "inc" ]; then
                    new_value=$((value + 1))
                else
                    new_value=$((value - 1))
                fi
                sleep 0.1
                echo "$new_value" > '"$counter_file"'
                echo "$1 $2: zapisano $new_value"
            ' dummy "$i" "$type"
        else
            value=$(cat "$counter_file")
            echo "$i $type: prebrano $value"
            if [ "$type" = "inc" ]; then
                new_value=$((value + 1))
            else
                new_value=$((value - 1))
            fi
            sleep 0.1
            echo "$new_value" >"$counter_file"
            echo "$i $type: zapisano $new_value"
        fi
    ) &
done

wait
echo "Končna vrednost: $(cat "$counter_file")"

#!/bin/bash

LOG="./temperatura.log"

temp=$(curl -s "wttr.in/Ljubljana?format=%t")

if [ -z "$temp" ]; then
    temp="/"
fi

timestamp=$(date '+%Y-%m-%d %H:%M:%S')

echo "$timestamp - Temperatura: $temp" >>"$LOG"

# za izvajanje skripte na vsake 30min sem dodal naslednjo vrstico v crontab (ukaz crontab -e)
# */30 * * * * /absolute_path_to/vreme.sh

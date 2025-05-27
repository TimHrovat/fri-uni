#!/bin/bash

SHOW_PID=1
SHOW_COMM=1
SHOW_MEM=1
SHOW_USER=1
SHOW_CPU=1

display_processes() {
    clear

    PID_WIDTH=7
    COMM_WIDTH=20
    MEM_WIDTH=6
    USER_WIDTH=12
    CPU_WIDTH=6

    output=""
    [[ $SHOW_PID -eq 1 ]] && output+=$(printf "%-*s" $PID_WIDTH "PID")
    [[ $SHOW_COMM -eq 1 ]] && output+=$(printf "%-*s" $COMM_WIDTH "COMMAND")
    [[ $SHOW_MEM -eq 1 ]] && output+=$(printf "%-*s" $MEM_WIDTH "%MEM")
    [[ $SHOW_USER -eq 1 ]] && output+=$(printf "%-*s" $USER_WIDTH "USER")
    [[ $SHOW_CPU -eq 1 ]] && output+=$(printf "%-*s" $CPU_WIDTH "%CPU")
    echo "$output"
    echo "---------------------------------------------------------------"

    ps_output=$(ps -eo pid,comm,%mem,user,%cpu | sort -k5 -nr | head -n 10)

    while read -r pid comm mem user cpu; do
        output=""
        [[ $SHOW_PID -eq 1 ]] && output+=$(printf "%-*s" $PID_WIDTH "$pid")
        [[ $SHOW_COMM -eq 1 ]] && output+=$(printf "%-*s" $COMM_WIDTH "${comm:0:17}...")
        [[ $SHOW_MEM -eq 1 ]] && output+=$(printf "%-*s" $MEM_WIDTH "$mem")
        [[ $SHOW_USER -eq 1 ]] && output+=$(printf "%-*s" $USER_WIDTH "$user")
        [[ $SHOW_CPU -eq 1 ]] && output+=$(printf "%-*s" $CPU_WIDTH "$cpu")

        echo "$output"
    done <<<"$ps_output"
}

display_help() {
    clear
    echo "Ukazi:"
    echo "  q - izhod iz programa (quit)"
    echo "  h - izpiše pomoč in čaka na pritisk tipke (help)"
    echo "  c - menjava vidnosti izpisa ukaza (command toggle display)"
    echo "  m - menjava vidnosti izpisa porabe pomnilnika (memory toggle display)"
    echo "  p - menjava vidnosti izpisa zasedenosti cpu (cpu toggle display)"
    echo "  u - menjava izpisa uporabnika (user toggle display)"
    echo
    echo "Pritisnite katerokoli tipko za nadaljevanje..."
    read -n 1 -r
}

while true; do
    display_processes

    read -r -t 1 -n 1 -s key

    case "$key" in
    q) exit 0 ;;
    h) display_help ;;
    c) SHOW_COMM=$((1 - SHOW_COMM)) ;;
    m) SHOW_MEM=$((1 - SHOW_MEM)) ;;
    p) SHOW_CPU=$((1 - SHOW_CPU)) ;;
    u) SHOW_USER=$((1 - SHOW_USER)) ;;
    esac

    unset key
done

#!/bin/bash

GROUP=${1:-student}

if ! getent group "$GROUP" >/dev/null; then
    groupadd "$GROUP"
fi

HOME_DIR="/home/$GROUP"
if [ ! -d "$HOME_DIR" ]; then
    mkdir -p "$HOME_DIR"
fi

while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi

    first_name=$(echo "$line" | awk '{print $1}')
    last_name=$(echo "$line" | awk '{print $2}')

    if [ -z "$first_name" ] || [ -z "$last_name" ]; then
        continue
    fi

    username_base="$(echo "${first_name:0:1}${last_name}" | tr '[:upper:]' '[:lower:]')"
    random_num=$(printf "%04d" $((RANDOM % 10000)))
    username="${username_base}${random_num}"

    full_name="$first_name $last_name"

    useradd -m -d "$HOME_DIR/$username" -g "$GROUP" -c "$full_name" "$username"
    echo "$username:$username" | chpasswd
done

echo "Vsi uporabniki uspešno ustvarjeni."

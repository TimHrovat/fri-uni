def naloga3(vhod: list, n: int) -> tuple[list, str]:
    m = 0
    while 2**m < n:
        m += 1

    k = n - m - 1

    received = vhod[:-1]
    overall_parity = vhod[-1]

    actual_parity = sum(received) % 2

    syndrome = []
    for i in range(m):
        parity_bit = 0
        for j in range(len(received)):
            if j+1 & (1 << i):
                parity_bit ^= received[j]

        syndrome.append(parity_bit)

    syndrome_val = 0
    for i, bit in enumerate(reversed(syndrome)):
        syndrome_val |= bit << i

    corrected = received.copy()

    if syndrome_val == 0 and actual_parity == overall_parity:
        pass
    elif syndrome_val != 0 and actual_parity != overall_parity:
        if 1 <= syndrome_val <= len(corrected):
            corrected[syndrome_val - 1] ^= 1
    else:
        return ([-1.0] * k, compute_crc(vhod))

    data_bits = []
    for i in range(len(corrected)):
        data_bits.append(corrected[i])

    if len(data_bits) > k:
        data_bits = data_bits[:k]

    return (data_bits, compute_crc(vhod))


def compute_crc(bits):
    polynomial = 0x9B
    crc = 0x00

    for bit in bits:
        feedback = ((crc >> 7) & 1) ^ bit

        crc = (crc << 1) & 0xFF

        if feedback:
            crc ^= polynomial

    return f"{crc:02X}"

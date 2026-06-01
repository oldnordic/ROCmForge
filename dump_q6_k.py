#!/usr/bin/env python3
"""
Debug Q6_K dequantization by dumping raw bytes and manually computing values
"""

import struct
import sys

def main():
    model_path = "/home/feanor/Projects/Memoria/models/qwen2-0.5b-instruct-q6_k.gguf"

    # Read GGUF file (simplified - just read first Q6_K tensor we can find)
    with open(model_path, 'rb') as f:
        data = f.read()

    # Find first Q6_K tensor by searching for the pattern
    # This is a hack - in reality we'd parse the GGUF format properly
    # For now, let's just look for a block that starts with reasonable Q6_K data

    print("File size: {} bytes".format(len(data)))

    # Look for Q6_K pattern: ql bytes should have some structure
    # We'll skip the GGUF header and look for tensor data

    # In a real implementation we'd parse the tensor info from GGUF
    # For debugging, let's just create a synthetic Q6_K block and verify our logic

    print("\nCreating synthetic Q6_K block for testing:")

    # Create a simple Q6_K block with known values
    ql = bytes(range(128))  # ql[0..128]
    qh = bytes([0xFF] * 64)  # qh[128..192] - all 1s in high bits
    scales = bytes([1] * 16)  # scales[192..208] - all 1s
    d_half = struct.pack('<e', 1.0)  # d[208..210] - scale = 1.0

    q6_k_block = ql + qh + scales + d_half

    print(f"Block size: {len(q6_k_block)} bytes (expected 210)")

    # Dump first 32 bytes
    print("\nFirst 32 bytes (ql array):")
    for i in range(0, 32, 16):
        print(" ".join(f"{b:02x}" for b in q6_k_block[i:i+16]))

    # Manually dequantize first 8 values
    print("\nManual dequantization of first 8 values:")

    # Parse d (fp16 at bytes 208-209)
    d_half_bytes = q6_k_block[208:210]
    d_val = struct.unpack('<e', d_half_bytes)[0]
    print(f"d = {d_val}")

    for i in range(8):
        group = i // 128
        pos_in_group = i % 128
        l_base = pos_in_group % 32
        quadrant = pos_in_group // 32
        is_ = l_base // 16
        scale_idx = group * 8 + is_ * 2 + quadrant

        scale = q6_k_block[192 + scale_idx]  # int8
        scale_signed = struct.unpack('<b', bytes([scale]))[0]

        ql_offset = group * 64 + l_base + quadrant * 32
        ql_byte = q6_k_block[ql_offset]

        qh_offset = 128 + group * 32 + l_base
        qh_byte = q6_k_block[qh_offset]

        is_low_half = 0 if quadrant < 2 else 1
        shift = 0 if is_low_half == 0 else 4
        qh_shift = ((quadrant % 2) * 2) if is_low_half == 0 else ((quadrant % 2) * 2 + 4)

        ql_4bits = (ql_byte >> shift) & 0x0F
        qh_2bits = (qh_byte >> qh_shift) & 0x03

        q_unsigned = ql_4bits | (qh_2bits << 4)
        q_signed = struct.unpack('<b', bytes([q_unsigned]))[0]

        # Apply formula: q = (unsigned_value) - 32
        q_final = q_signed - 32 if q_signed >= 0 else q_signed + 256 - 32

        value = d_val * scale_signed * q_final

        print(f"i={i}, group={group}, l_base={l_base}, quadrant={quadrant}, "
              f"is_={is_}, scale_idx={scale_idx}, scale={scale_signed}, "
              f"ql_4bits={ql_4bits}, qh_2bits={qh_2bits}, q={q_final}, value={value:.4f}")

if __name__ == "__main__":
    main()

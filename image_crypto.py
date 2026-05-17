import numpy as np
import cv2
import hashlib
from PIL import Image
import matplotlib.pyplot as plt

# =========================================================
# DNA RULES
# =========================================================

DNA_ENCODE_RULE = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'T'
}

DNA_DECODE_RULE = {v: k for k, v in DNA_ENCODE_RULE.items()}

DNA_XOR_TABLE = {
    ('A','A'):'A', ('A','C'):'C', ('A','G'):'G', ('A','T'):'T',
    ('C','A'):'C', ('C','C'):'A', ('C','G'):'T', ('C','T'):'G',
    ('G','A'):'G', ('G','C'):'T', ('G','G'):'A', ('G','T'):'C',
    ('T','A'):'T', ('T','C'):'G', ('T','G'):'C', ('T','T'):'A'
}

# =========================================================
# HASH GENERATION
# =========================================================

def generate_hash(image):
    """
    Generate SHA-256 hash from image
    """
    img_bytes = image.tobytes()
    hash_hex = hashlib.sha256(img_bytes).hexdigest()

    print("SHA-256 Hash:")
    print(hash_hex)

    return hash_hex


# =========================================================
# FIBONACCI Q-MATRIX
# =========================================================

def fibonacci(n):
    """
    Fibonacci sequence
    """
    if n <= 1:
        return 1

    f1, f2 = 1, 1

    for _ in range(2, n + 1):
        f1, f2 = f2, f1 + f2

    return f2


def fibonacci_q_matrix_seed(hash_hex):
    """
    Generate initial condition from Fibonacci Q-Matrix
    """

    # Ambil 6 bagian hash
    segments = [
        hash_hex[0:10],
        hash_hex[10:20],
        hash_hex[20:30],
        hash_hex[30:40],
        hash_hex[40:50],
        hash_hex[50:64]
    ]

    seeds = []

    for seg in segments:
        value = int(seg, 16)

        # Fibonacci index
        n = (value % 20) + 5

        Fn = fibonacci(n)
        Fn1 = fibonacci(n + 1)

        # Q-matrix power
        Qn = np.array([
            [Fn1, Fn],
            [Fn, fibonacci(n - 1)]
        ])

        # normalisasi
        seed = (np.sum(Qn) % 1000) / 1000.0

        seeds.append(seed)

    return seeds


# =========================================================
# 6D HYPERCHAOTIC SYSTEM (RK4)
# =========================================================

def chaotic_equations(state, a, b, c, d, e, r):

    x1, x2, x3, x4, x5, x6 = state

    dx1 = a*(x2 - x1) + x4 - x5 - x6

    dx2 = c*x1 - x2 - x1*x3

    dx3 = -b*x3 + x1*x2

    dx4 = d*x4 - x2*x3

    dx5 = e*x6 + x3*x2

    dx6 = r*x1

    return np.array([
        dx1,
        dx2,
        dx3,
        dx4,
        dx5,
        dx6
    ], dtype=np.float64)


# =========================================================

def rk4_step(state, dt, a, b, c, d, e, r):

    k1 = chaotic_equations(state, a, b, c, d, e, r)

    k2 = chaotic_equations(
        state + 0.5 * dt * k1,
        a, b, c, d, e, r
    )

    k3 = chaotic_equations(
        state + 0.5 * dt * k2,
        a, b, c, d, e, r
    )

    k4 = chaotic_equations(
        state + dt * k3,
        a, b, c, d, e, r
    )

    next_state = state + (dt / 6.0) * (
        k1 + 2*k2 + 2*k3 + k4
    )

    return next_state


# =========================================================

def hyperchaotic_6d(
        seeds,
        usable_size=36000,
        discard=1000,
        a=10,
        b=8/3,
        c=28,
        d=-1,
        e=8,
        r=3,
        dt=0.01
    ):

    # =====================================
    # Initial state
    # =====================================

    state = np.array(seeds, dtype=np.float64)

    seq1 = []
    seq2 = []
    seq3 = []
    seq4 = []
    seq5 = []
    seq6 = []

    total_iteration = usable_size + discard

    # =====================================
    # RK4 ITERATION
    # =====================================

    for i in range(total_iteration):

        state = rk4_step(
            state,
            dt,
            a, b, c, d, e, r
        )

        # =================================
        # Numerical stabilization
        # =================================

        state = np.clip(state, -50, 50)

        # =================================
        # Normalize
        # =================================

        normalized = np.mod(
            np.abs(state * 1e6),
            1.0
        )

        # =================================
        # Remove transient
        # =================================

        if i >= discard:

            seq1.append(normalized[0])
            seq2.append(normalized[1])
            seq3.append(normalized[2])
            seq4.append(normalized[3])
            seq5.append(normalized[4])
            seq6.append(normalized[5])

    return (
        np.array(seq1),
        np.array(seq2),
        np.array(seq3),
        np.array(seq4),
        np.array(seq5),
        np.array(seq6)
    )


# =========================================================
# DNA ENCODING
# =========================================================

def byte_to_dna(byte):
    """
    Convert 8-bit byte to DNA sequence
    """

    binary = format(byte, '08b')

    dna = ""

    for i in range(0, 8, 2):
        bits = binary[i:i+2]
        dna += DNA_ENCODE_RULE[int(bits, 2)]

    return dna


def dna_to_byte(dna):
    """
    Convert DNA back to byte
    """

    binary = ""

    for base in dna:
        binary += format(DNA_DECODE_RULE[base], '02b')

    return int(binary, 2)


def dna_xor_byte(byte1, byte2):

    dna1 = byte_to_dna(byte1)
    dna2 = byte_to_dna(byte2)

    result = ""

    for a, b in zip(dna1, dna2):
        result += DNA_XOR_TABLE[(a, b)]

    return dna_to_byte(result)


# =========================================================
# KEY GENERATION
# =========================================================

def generate_key_stream(sequences, size):

    key = np.zeros(size, dtype=np.uint8)

    combined = (
        sequences[0] * 0.17 +
        sequences[1] * 0.19 +
        sequences[2] * 0.23 +
        sequences[3] * 0.29 +
        sequences[4] * 0.31 +
        sequences[5] * 0.37
    )

    combined = np.nan_to_num(combined)

    for i in range(size):

        value = combined[i]

        key[i] = int(
            np.floor(
                (value * 1e15) % 256
            )
        )

    return key

# =========================================================
# DNA XOR ENCRYPTION
# =========================================================

def dna_encrypt_channel(channel, key):

    flat = channel.flatten()

    encrypted = np.zeros_like(flat)

    for i in range(len(flat)):
        encrypted[i] = dna_xor_byte(flat[i], key[i])

    return encrypted.reshape(channel.shape)


def dna_decrypt_channel(channel, key):
    return dna_encrypt_channel(channel, key)


# =========================================================
# SCRAMBLING
# =========================================================

def scramble_channel(channel, seqA, seqB):

    flat = channel.flatten()

    index = np.argsort(seqA + seqB)

    scrambled = flat[index]

    return scrambled.reshape(channel.shape), index


def descramble_channel(channel, index):

    flat = channel.flatten()

    recovered = np.zeros_like(flat)

    recovered[index] = flat

    return recovered.reshape(channel.shape)


# =========================================================
# MAIN ENCRYPTION
# =========================================================

def encrypt_image(image_path):

    # Load image
    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    size = h * w

    # =====================================================
    # HASH
    # =====================================================

    hash_hex = generate_hash(img)

    # =====================================================
    # FIBONACCI Q-MATRIX
    # =====================================================

    seeds = fibonacci_q_matrix_seed(hash_hex)

    print("\nInitial Seeds:")
    print(seeds)

    # =====================================================
    # HYPERCHAOTIC SEQUENCE
    # =====================================================

    sequences = hyperchaotic_6d(seeds, size)

    # =====================================================
    # KEY GENERATION
    # =====================================================

    key = generate_key_stream(sequences, size)

    # =====================================================
    # SPLIT RGB
    # =====================================================

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # =====================================================
    # DNA XOR
    # =====================================================

    R_enc = dna_encrypt_channel(R, key)
    G_enc = dna_encrypt_channel(G, key)
    B_enc = dna_encrypt_channel(B, key)

    # =====================================================
    # SCRAMBLING
    # =====================================================

    R_scr, idxR = scramble_channel(
        R_enc,
        sequences[0],
        sequences[1]
    )

    G_scr, idxG = scramble_channel(
        G_enc,
        sequences[2],
        sequences[3]
    )

    B_scr, idxB = scramble_channel(
        B_enc,
        sequences[4],
        sequences[5]
    )

    encrypted = np.stack([R_scr, G_scr, B_scr], axis=2)

    return encrypted, key, (idxR, idxG, idxB), sequences


# =========================================================
# MAIN DECRYPTION
# =========================================================

def decrypt_image(encrypted, key, indices):

    idxR, idxG, idxB = indices

    R = encrypted[:, :, 0]
    G = encrypted[:, :, 1]
    B = encrypted[:, :, 2]

    # =====================================================
    # DESCRAMBLE
    # =====================================================

    R_desc = descramble_channel(R, idxR)
    G_desc = descramble_channel(G, idxG)
    B_desc = descramble_channel(B, idxB)

    # =====================================================
    # DNA XOR DECRYPTION
    # =====================================================

    R_dec = dna_decrypt_channel(R_desc, key)
    G_dec = dna_decrypt_channel(G_desc, key)
    B_dec = dna_decrypt_channel(B_desc, key)

    decrypted = np.stack([R_dec, G_dec, B_dec], axis=2)

    return decrypted


# =========================================================
# VISUALIZATION
# =========================================================

def show_results(original, encrypted, decrypted):

    plt.figure(figsize=(15, 5))

    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(encrypted)
    plt.title("Encrypted")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(decrypted)
    plt.title("Decrypted")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":

    image_path = "test.png"

    # Load original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Encrypt
    encrypted, key, indices, sequences = encrypt_image(image_path)

    # Decrypt
    decrypted = decrypt_image(
        encrypted,
        key,
        indices
    )

    # Save result
    Image.fromarray(encrypted).save("encrypted.png")
    Image.fromarray(decrypted).save("decrypted.png")

    # Visualization
    show_results(
        original,
        encrypted,
        decrypted
    )

import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# PARAMETER
# =========================
a, b, c, d, e, r = 10, 8/3, 28, -1, 8, 3
dt = 0.01

Q = np.array([[89, 55],
              [55, 34]])

Q_inv = np.array([[34, -55],
                  [-55, 89]])

# =========================
# HYPERCHAOTIC SYSTEM
# =========================
def deriv(x):
    x1, x2, x3, x4, x5, x6 = x
    
    dx1 = a*(x2 - x1) + x4 - x5 - x6
    dx2 = c*x1 - x2 - x1*x3
    dx3 = -b*x3 + x1*x2
    dx4 = d*x4 - x2*x3
    dx5 = e*x6 + x3*x2
    dx6 = r*x1
    
    return np.array([dx1, dx2, dx3, dx4, dx5, dx6])

def rk4_step(x, dt):
    k1 = deriv(x)
    k2 = deriv(x + 0.5*dt*k1)
    k3 = deriv(x + 0.5*dt*k2)
    k4 = deriv(x + dt*k3)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# =========================
# CHAOTIC SEQUENCE
# =========================
def generate_sequence(x0, length, discard=1000):
    x = np.array(x0)
    seq = []

    total_iter = discard + length//3 + 10

    for i in range(total_iter):
        x = rk4_step(x, dt)
        if i >= discard:
            seq.extend([x[0], x[2], x[4]])

    return np.array(seq[:length])

# =========================
# CONFUSION
# =========================
def confusion(P, seq):
    S = np.argsort(seq)
    return P[S], S

def deconfusion(R, S):
    P = np.zeros_like(R)
    P[S] = R
    return P

# =========================
# DIFFUSION
# =========================
def diffusion(img):
    M, N = img.shape
    out = np.zeros_like(img, dtype=np.int32)

    for i in range(0, M-1, 2):
        for j in range(0, N-1, 2):
            block = img[i:i+2, j:j+2].astype(np.int32)
            out[i:i+2, j:j+2] = np.dot(block, Q) % 256

    return out.astype(np.uint8)

def inverse_diffusion(img):
    M, N = img.shape
    out = np.zeros_like(img, dtype=np.int32)

    for i in range(0, M-1, 2):
        for j in range(0, N-1, 2):
            block = img[i:i+2, j:j+2].astype(np.int32)
            out[i:i+2, j:j+2] = np.dot(block, Q_inv) % 256

    return out.astype(np.uint8)

# =========================
# ENCRYPTION
# =========================
def encrypt(image, rounds=2):
    I = image.copy()
    M, N = I.shape
    keys = []

    for _ in range(rounds):
        P = I.flatten()

        # initial condition (disederhanakan biar stabil)
        x1_init = (np.sum(P) % 256) / 256
        x0 = [x1_init]*6

        seq = generate_sequence(x0, M*N)

        R, S = confusion(P, seq)
        keys.append(S)

        R_mat = R.reshape(M, N)
        I = diffusion(R_mat)

    return I, keys

# =========================
# DECRYPTION
# =========================
def decrypt(cipher, keys, rounds=2):
    I = cipher.copy()
    M, N = I.shape

    for r in reversed(range(rounds)):
        S = keys[r]

        # inverse diffusion
        D = inverse_diffusion(I)

        # reshape + flatten
        R = D.flatten()

        # deconfusion
        P = deconfusion(R, S)

        I = P.reshape(M, N)

    return I

# =========================
# MAIN
# =========================
img = cv2.imread("test.png", 0)

# pastikan ukuran genap
img = img[:img.shape[0]//2*2, :img.shape[1]//2*2]

encrypted, keys = encrypt(img)
decrypted = decrypt(encrypted, keys)

# =========================
# VISUALISASI
# =========================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(encrypted, cmap='gray')
plt.title("Encrypted")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(decrypted, cmap='gray')
plt.title("Decrypted")
plt.axis('off')

plt.tight_layout()
plt.show()

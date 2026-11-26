import json
import math
import numpy as np
from PIL import Image

EPSILON = 1
MAX_ITERATIONS = 100


def load_image(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.int32)


def save_image(array, path):
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L").save(path)


def pad_image(image, block_h, block_w):
    h, w = image.shape
    pad_h = (block_h - h % block_h) % block_h
    pad_w = (block_w - w % block_w) % block_w
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded


def image_to_blocks(image, block_h, block_w):
    h, w = image.shape
    blocks = []
    for r in range(0, h, block_h):
        for c in range(0, w, block_w):
            blocks.append(image[r:r + block_h, c:c + block_w].reshape(-1))
    return np.array(blocks, dtype=np.float64)


def blocks_to_image(blocks, padded_h, padded_w, block_h, block_w):
    image = np.zeros((padded_h, padded_w), dtype=np.int32)
    idx = 0
    for r in range(0, padded_h, block_h):
        for c in range(0, padded_w, block_w):
            image[r:r + block_h, c:c + block_w] = np.array(blocks[idx]).reshape(block_h, block_w)
            idx += 1
    return image


def calculate_distances(blocks, codebook):
    distances = np.empty((len(blocks), len(codebook)))
    chunk = 2048
    for start in range(0, len(blocks), chunk):
        part = blocks[start:start + chunk]
        distances[start:start + chunk] = np.abs(
            part[:, None, :] - codebook[None, :, :]
        ).sum(axis=2)
    return distances


def find_nearest(blocks, codebook):
    distances = calculate_distances(blocks, codebook)
    return np.argmin(distances, axis=1)


def update_centroids(blocks, labels, codebook):
    new_codebook = codebook.copy()
    for i in range(len(codebook)):
        members = blocks[labels == i]
        if len(members) > 0:
            new_codebook[i] = members.mean(axis=0)
    return new_codebook


def split_codebook(codebook, target_size):
    new_codebook = []
    for vector in codebook:
        new_codebook.append(vector - EPSILON)
        new_codebook.append(vector + EPSILON)
    new_codebook = np.array(new_codebook)

    if len(new_codebook) > target_size:
        extra = target_size - len(codebook)
        result = []
        for i, vector in enumerate(codebook):
            if i < extra:
                result.append(vector - EPSILON)
                result.append(vector + EPSILON)
            else:
                result.append(vector)
        new_codebook = np.array(result)

    return new_codebook


def run_lbg_iterations(blocks, codebook):
    labels = find_nearest(blocks, codebook)
    for _ in range(MAX_ITERATIONS):
        codebook = update_centroids(blocks, labels, codebook)
        new_labels = find_nearest(blocks, codebook)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return codebook


def generate_codebook(blocks, codebook_size):
    codebook = blocks.mean(axis=0).reshape(1, -1)

    while len(codebook) < codebook_size:
        codebook = split_codebook(codebook, codebook_size)
        codebook = run_lbg_iterations(blocks, codebook)

    codebook = np.clip(np.rint(codebook), 0, 255)
    return codebook


def pack_bits(bit_string):
    remainder = len(bit_string) % 8
    if remainder != 0:
        padding_count = 8 - remainder
    else:
        padding_count = 0

    bit_string += "0" * padding_count

    bytes_list = bytearray()
    for i in range(0, len(bit_string), 8):
        bytes_list.append(int(bit_string[i:i + 8], 2))

    return bytes(bytes_list), padding_count


def unpack_bits(binary_data, padding_count):
    bit_string = "".join(format(byte, "08b") for byte in binary_data)
    if padding_count > 0:
        bit_string = bit_string[:-padding_count]
    return bit_string


def calculate_mse(original, reconstructed):
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_compression_ratio(height, width, block_h, block_w, codebook_size, bit_len):
    original_bits = height * width * 8
    padded_h = height + (block_h - height % block_h) % block_h
    padded_w = width + (block_w - width % block_w) % block_w
    num_blocks = (padded_h * padded_w) // (block_h * block_w)
    labels_bits = num_blocks * bit_len
    codebook_bits = codebook_size * block_h * block_w * 8
    compressed_bits = codebook_bits + labels_bits
    return original_bits, labels_bits, codebook_bits, compressed_bits, original_bits / compressed_bits


def compress(input_image, codebook_file, compressed_file, block_h, block_w, codebook_size):
    image = load_image(input_image)
    height, width = image.shape

    padded = pad_image(image, block_h, block_w)
    padded_h, padded_w = padded.shape
    blocks = image_to_blocks(padded, block_h, block_w)

    codebook_size = min(codebook_size, len(blocks))
    codebook = generate_codebook(blocks, codebook_size)

    labels = find_nearest(blocks, codebook)

    if codebook_size == 1:
        bit_len = 1
    else:
        bit_len = math.ceil(math.log2(codebook_size))

    bit_string = "".join(format(int(label), f"0{bit_len}b") for label in labels)
    binary_data, padding_count = pack_bits(bit_string)

    codebook_data = {
        "block_h": block_h,
        "block_w": block_w,
        "codebook_size": int(codebook_size),
        "codebook": codebook.astype(int).tolist(),
    }
    with open(codebook_file, "w") as f:
        json.dump(codebook_data, f)

    header = {
        "original_height": height,
        "original_width": width,
        "padding_count": padding_count,
        "bit_len": bit_len,
        "num_blocks": len(blocks),
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with open(compressed_file, "wb") as f:
        f.write(len(header_bytes).to_bytes(4, byteorder="big"))
        f.write(header_bytes)
        f.write(binary_data)

    reconstructed_blocks = codebook[labels]
    reconstructed = blocks_to_image(reconstructed_blocks, padded_h, padded_w, block_h, block_w)
    reconstructed = reconstructed[:height, :width]
    mse = calculate_mse(image, reconstructed)

    original_bits, labels_bits, codebook_bits, compressed_bits, ratio = calculate_compression_ratio(
        height, width, block_h, block_w, codebook_size, bit_len
    )

    print("Compressed ->", codebook_file, "+", compressed_file)
    print("Image size              :", height, "x", width, "(padded to", padded_h, "x", padded_w, ")")
    print("Block size              :", block_h, "x", block_w)
    print("Number of blocks        :", len(blocks))
    print("Codebook vectors        :", codebook_size, "->", bit_len, "bits per label")
    print("Original size           :", original_bits, "bits")
    print("Labels size             :", labels_bits, "bits")
    print("Codebook size           :", codebook_bits, "bits")
    print("Total compressed size   :", compressed_bits, "bits")
    print("Compression Ratio       : %.2f : 1" % ratio)
    print("Mean Square Error (MSE) : %.4f" % mse)


def decompress(codebook_file, compressed_file, decompressed_file):
    with open(codebook_file, "r") as f:
        codebook_data = json.load(f)

    block_h = codebook_data["block_h"]
    block_w = codebook_data["block_w"]
    codebook = np.array(codebook_data["codebook"])

    with open(compressed_file, "rb") as f:
        header_length = int.from_bytes(f.read(4), byteorder="big")
        header = json.loads(f.read(header_length).decode("utf-8"))
        binary_data = f.read()

    height = header["original_height"]
    width = header["original_width"]
    padding_count = header["padding_count"]
    bit_len = header["bit_len"]
    num_blocks = header["num_blocks"]

    padded_h = height + (block_h - height % block_h) % block_h
    padded_w = width + (block_w - width % block_w) % block_w

    bit_string = unpack_bits(binary_data, padding_count)

    blocks = []
    for i in range(num_blocks):
        label = int(bit_string[i * bit_len:(i + 1) * bit_len], 2)
        blocks.append(codebook[label])

    image = blocks_to_image(blocks, padded_h, padded_w, block_h, block_w)
    image = image[:height, :width]
    save_image(image, decompressed_file)

    print("Decompressed ->", decompressed_file)


if __name__ == "__main__":
    while True:
        print("\n1) Compress")
        print("2) Decompress")
        print("3) Exit")
        choice = input("Choose: ").strip()

        try:
            if choice == "1":
                image_path = input("Image to compress: ").strip()
                block_h = int(input("Block (vector) height: "))
                block_w = int(input("Block (vector) width: "))
                codebook_size = int(input("Number of vectors in codebook: "))
                compress(image_path, "codebook.json", "compressed.bin", block_h, block_w, codebook_size)

            elif choice == "2":
                decompress("codebook.json", "compressed.bin", "decompressed.png")

            elif choice == "3":
                print("Exiting...")
                break

            else:
                print("Invalid choice! Please enter 1, 2, or 3.")

        except FileNotFoundError:
            print("Error: File not found! Check the file path and try again.")
        except Exception as e:
            print(f"Error: {e}")
import argparse
import hashlib

import numpy as np

from prediction import MODEL_FILES, run_prediction

MODEL_SHA256 = {
    "upward": "52fbda0c2fac68d5e479674499a746661974f9185c5377b6706ed9d454928ac3",
    "downward": "da971ff7a56f4404bdaa18e44ab3465d64c05b259c36bcc22d90ab22f8fbc4f5",
}

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def mean_correlation(predicted, observed):
    numerator = np.sum(predicted * observed, axis=1)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(observed, axis=1)
    return float(np.mean(numerator / denominator))

def verify(direction, device, check_hash=True):
    if check_hash:
        actual_hash = sha256_file(MODEL_FILES[direction])
        if actual_hash != MODEL_SHA256[direction]:
            raise ValueError(direction + " model checksum does not match")

    example, predicted_a, predicted_v, _ = run_prediction(direction, device)
    expected_a = example["expected_acceleration"]
    expected_v = example["expected_velocity"]

    max_a_error = float(np.max(np.abs(predicted_a - expected_a)))
    max_v_error = float(np.max(np.abs(predicted_v - expected_v)))
    passed = np.allclose(predicted_a, expected_a, atol=2e-5, rtol=2e-4)
    passed = passed and np.allclose(predicted_v, expected_v, atol=2e-5, rtol=2e-4)

    print(direction, "record", int(example["record_id"][0]))
    print("  maximum acceleration error:", max_a_error)
    print("  maximum velocity error:    ", max_v_error)
    print(
        "  acceleration correlation:  ",
        mean_correlation(predicted_a, example["observed_acceleration"]),
    )
    print(
        "  velocity correlation:      ",
        mean_correlation(predicted_v, example["observed_velocity"]),
    )
    print("  passed:", passed)
    if not passed:
        raise AssertionError(direction + " prediction does not match the reference")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("both", "upward", "downward"), default="both")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    directions = ("upward", "downward") if args.direction == "both" else (args.direction,)
    for selected_direction in directions:
        verify(selected_direction, args.device, check_hash=not args.skip_hash)

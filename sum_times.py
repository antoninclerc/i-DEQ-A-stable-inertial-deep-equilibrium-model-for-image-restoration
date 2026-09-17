import re

def sum_iteration_times(file_path: str) -> float:
    """
    Read a file containing a line like:
    Time per iteration: [0.5, 1.2, 3.4, ...]

    Extract the numbers and return their sum.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the list inside brackets
    match = re.search(r"Time per iteration:\s*\[(.*?)\]", content, re.S)
    if not match:
        raise ValueError("No valid 'Time per iteration' list found in file.")

    numbers_str = match.group(1)

    # Convert to floats
    numbers = [float(x) for x in numbers_str.split(",") if x.strip()]

    return sum(numbers)


if __name__ == "__main__":
    file_path = "Unrolling_comparison/DEQs/Rician/prox_01/test_times.txt"  # change this to your file
    total_time = sum_iteration_times(file_path)
    print("Total time:", total_time)
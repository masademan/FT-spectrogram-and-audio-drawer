import os
import time
from fft_utils import (
    lib_rfft,
    custom_rfft,
    custom_rdft,
    AudioSignal,
)

def print_and_write(text="", file=None):
    print(text)

    if not file:
        return
    
    file.write(text + "\n")

def main() -> None:

    filename = "yippeee!!! daniel.mp3"
    with open(os.path.join("Data", "Compute time TXTs", f"Compute time of '{filename}'.txt"), "w") as file:
        audio_signal = AudioSignal.from_audio_file(filename, target_sample_rate=None)
        functions = [lib_rfft, custom_rfft, custom_rdft]

        TAB = "  "
        print_and_write(f"Testing computing time for file '{filename}' on functions:", file)
        for func in functions:
            print_and_write(f"{TAB}{func.__name__}", file)
        print_and_write(file=file)

        trials_per_func = 5
        for func in functions:
            total_time = 0

            for _ in range(trials_per_func):
                start = time.perf_counter()
                func(audio_signal)
                end = time.perf_counter()
                total_time += end - start
            
            total_time /= trials_per_func

            print_and_write(f"Compute time of {func.__name__}: {total_time:.2f} s", file)


if __name__ == "__main__":
    main()
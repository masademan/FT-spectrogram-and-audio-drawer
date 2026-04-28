import os
import librosa
import functools
import numpy as np
import pandas as pd
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt
from collections.abc import Callable
from matplotlib.colors import Colormap
from utils import (
    get_color_map,
    is_power_of_two,
    smallest_power_of_two_greater_than_n,
)

@functools.lru_cache(maxsize=32)
def get_twiddle_factors(n: int) -> np.ndarray:
    return np.exp(-2j * np.pi * np.arange(n//2) / n)

class Note:
    """
    Data for a musical note in Western music with piano notation

    This class holds data of which note it is, as well as amplitude, phase, delay, and duration
    for .wav audio file creation

    Attributes:
        note (tuple[str, str, int]): Tuple containing the note name (A, B, C, etc.), whether it's sharp, neutral, or flat (#, -, b), and the octave number (0, 1, ..., 7, 8)
        amplitude (float): Measure of amplitude (not in dB), default is 1
        phase (float): Amount of phase shift to apply to the sin function when reconstructing audio with notes
        delay (float): Delay in the audio to play this note
        duration (float): Duration to play this not in the audio file, float("inf") makes the note play until the end
    """
    
    def __init__(self, note: tuple[str, str, int], amplitude: float = 1, phase: float = 0, delay: float = 0, duration: float = float("inf")) -> None:
        """
        Initializes the Note object

        Args:
            note (tuple[str, str, int]): Tuple containing the note name (A, B, C, etc.), whether it's sharp, neutral, or flat (#, -, b), and the octave number (0, 1, ..., 7, 8)
            amplitude (float): Measure of amplitude (not in dB), default is 1
            phase (float): Amount of phase shift to apply to the sin function when reconstructing audio with notes
            delay (float): Delay in the audio to play this note
            duration (float): Duration to play this not in the audio file, float("inf") makes the note play until the end
        """
        self.note = self.normalize_note((note[0].upper(), note[1].lower(), note[2]))
        self.amplitude = amplitude
        self.phase = phase
        self.delay = delay
        self.duration = duration
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Note):
            return NotImplemented
        
        return self.note == other.note and self.amplitude == other.amplitude and self.phase == other.phase and self.delay == other.delay and self.duration == other.duration
    
    def __hash__(self):
        return hash((self.note, self.amplitude, self.phase, self.delay, self.duration))
    
    def __str__(self):
        return f"{self.note[0]}{self.note[1]}{self.note[2]}"
    
    @classmethod
    def from_string(cls, str_note: str, amplitude: float = 1, phase: float = 0, delay: float = 0, duration: float = float("inf")) -> Note:
        """
        Rather than using a tuple for initialization, you can use a string, which can be easier to make

        Args:
            str_note (str): String of length 3, containing the note name (A, B, C, etc.), whether it's sharp, neutral, or flat (#, -, b), and the octave number (0, 1, ..., 7, 8)
            amplitude (float): Measure of amplitude (not in dB), default is 1
            phase (float): Amount of phase shift to apply to the sin function when reconstructing audio with notes
            delay (float): Delay in the audio to play this note
            duration (float): Duration to play this not in the audio file, float("inf") makes the note play until the end
        
        Returns:
            Note: The initialized Note object from the string
        """
        note = cls.normalize_note((str_note[0].upper(), str_note[1].lower(), int(str_note[2])))
        return cls(note, amplitude, phase, delay, duration)
    
    @classmethod
    def normalize_note(cls, note: tuple[str, str, int]) -> tuple[str, str, int]:
        """
        Takes a tupled note, and normalizes it
        Converts flats into sharps and makes sure there aren't any sharps or flats pointing to a normally neutral note

        Args:
            note (tuple[str, str, int]): Tuple containing the note name (A, B, C, etc.), whether it's sharp, neutral, or flat (#, -, b), and the octave number (0, 1, ..., 7, 8)
        
        Returns:
            tuple[str, str, int]: The tuple of a normalized note, which is ready to feed into the Note initialization
        """
        if note[1] == "-" or (note[0] not in ["E", "B"] and note[1] == "#"):
            return note
        
        new_note = list(note)

        if note[1] == "#":
            new_note[0] = "F" if note[0] == "E" else "C"
            new_note[1] = "-"
        if note[1] == "b":
            new_note_letter = chr(ord(note[0]) - 1)
            if new_note_letter == "@":
                new_note_letter = "G"
            new_note[0] = new_note_letter
            new_note[1] = "#"

            if note[0] in ["C", "F"]:
                new_note[1] = "-"

        return new_note

    def get_freq(self) -> float:
        """
        Get the frequency in Hz of the current note
        Based on a table of notes in the 4th octave and then scaled up or down to get to the correct octave

        Returns:
            float: The frequency of the note in Hz
        """
        current_octave = 4
        note_to_freq = { # Octave 4 - Freq * 100 to store as ints
            ("C", "-"): 261_63,
            ("C", "#"): 277_18,
            ("D", "-"): 293_66,
            ("D", "#"): 311_13,
            ("E", "-"): 329_63,
            ("F", "-"): 349_23,
            ("F", "#"): 369_99,
            ("G", "-"): 392_00,
            ("G", "#"): 415_30,
            ("A", "-"): 440_00,
            ("A", "#"): 466_16,
            ("B", "-"): 493_88,
        }

        current_freq = note_to_freq[self.note[:2]]
        octave_offset = self.note[2] - current_octave

        return current_freq * 2**(octave_offset) / 100
    
    def set_amplitude(self, new_amplitude: float) -> None:
        """
        Change the current amplitude to a new value

        Args:
            new_amplitude (float): New value for self.amplitude
        """
        self.amplitude = new_amplitude

    def set_phase(self, new_phase: float) -> None:
        """
        Change the current phase to a new value

        Args:
            new_phase (float): New value for self.phase
        """
        self.phase = new_phase

    def set_delay(self, new_delay: float) -> None:
        """
        Change the current delay to a new value

        Args:
            new_delay (float): New value for self.delay
        """
        self.delay = new_delay

    def set_duration(self, new_duration: float) -> None:
        """
        Change the current duration to a new value

        Args:
            new_duration (float): New value for self.duration
        """
        self.duration = new_duration

    def play_note(self, sample_rate: float = 2000, duration_in: float | None = None, amplitude_in: float | None = None) -> None:
        """
        Plays the note stored and the duration and amplitude given
        The amplitude and duration can be overridden to get a different effect

        Args:
            sample_rate (float): The sample_rate of the audio playback, the lower the sample_rate, the lower the quality. But the higher the sample_rate, the higher the quality, up to a certain point
            duration_in (float | None): None by default, but giving it a value will override the saved duration value
            amplitude_in (float | None): None by default, but giving it a value will override the saved amplitude value
        """
        duration = duration_in if duration_in is not None else self.duration
        amplitude = amplitude_in if amplitude_in is not None else self.amplitude

        total_samples = int(sample_rate * duration)
        time = np.arange(total_samples) / sample_rate

        amplitude_over_time = amplitude * np.sin(2 * np.pi * self.get_freq() * time)

        normalized_amplitude = amplitude_over_time / np.max(np.abs(amplitude_over_time))
        audio_data = normalized_amplitude.astype(np.float32)

        sd.play(audio_data, sample_rate)
        sd.wait()

class AudioSignal:
    """
    A Pandas dataframe of the audio signal given as a discreet function of amplitudes over time

    This AudioSignal can be made in different ways, such as directly through the Pandas dataframe,
    importing a .csv of the amplitude vs time, a list of notes (Note, in string form or as the class),
    or as the Fourier Transform output data (dict or FourierTransformData class)

    Attributes:
        time_vs_amplitude (pandas.Dataframe): A Pandas dataframe with one column named "Time" and another "Amplitude", holding the amplitudes at each given time
    """

    def __init__(self, audio_df: pd.DataFrame, sample_rate: int | None = None) -> None:
        """
        Initializes the AudioSignal object

        Args:
            time_vs_amplitude (pandas.Dataframe): A Pandas dataframe with one column named "Time" and another "Amplitude", holding the amplitudes at each given time
        """
        needed_columns = ["Time", "Amplitude"]
        for column in needed_columns:
            if column not in audio_df.columns.tolist():
                raise ValueError
        self.time_vs_amplitude = audio_df
        self.time_vs_amplitude["Time"] -= self.time_vs_amplitude["Time"].min()
        self.time_vs_amplitude = self.time_vs_amplitude.sort_values(by="Time")

    def __eq__(self, other: AudioSignal) -> bool:
        return self.time_vs_amplitude.equals(other.time_vs_amplitude)
    
    @classmethod
    def from_csv_file(cls, file_name: str = "secret_chord.csv", use_absolute_path: bool = False) -> AudioSignal:
        """
        Reads a .csv holding the amplitude vs time data of an audio file,
        and turns it into a Pandas dataframe to initialize the AudioSignal object

        Args:
            file_name (str): The file name of the .csv file to read, the .csv should be in a specific folder and the program will search in that directory\n
                             The only thing you need to do it to just input the file name, nothing else
            use_absolute_path (bool): Whether to load from the Data directory, or to use a different absolute path
        
        Returns:
            AudioSignal: The AudioSignal object from the .csv file
        """
        if not use_absolute_path:
            if not file_name.endswith(".csv"):
                file_name = file_name + ".csv"
            file_name = os.path.join("Data", "CSVs", file_name)
        
        audio_df = pd.read_csv(file_name)
        audio_df.rename(columns={"Time (s)": "Time"}, inplace=True)
        return cls(audio_df)
    
    @classmethod
    def from_audio_file(cls, audio_file: str, target_sample_rate: float | None = 2000, use_absolute_path: bool = False) -> AudioSignal:
        """
        Reads a .mp3 or .wav audio file to create a Pandas dataframe of the
        amplitude of the audio over time

        The audio's sample_rate can be downcast to the inputted value, but 
        setting target_sample_rate to None will read the audio file using the native sample rate
        rather than down casting it

        Args:
            audio_file (str): The file name of the .mp3 or .wav file to read, the audio file should be in a specific folder and the program will search in that directory\n
                              The only thing you need to do it to just input the file name, nothing else
            target_sample_rate (float | None): The sample rate to downcast the audio file to, but leaving it at None will use the native sample rate of the audio file
            use_absolute_path (bool): Whether to load from the Data directory, or to use a different absolute path

        Returns:
            AudioSignal: The AudioSignal object from the .mp3 or .wav audio file
        """
        if not use_absolute_path:
            audio_file_type = audio_file.split(".")[-1]
            if audio_file_type not in ["mp3", "wav"]:
                audio_file_path = os.path.join("Data", "MP3 or WAV", "WAV", audio_file + ".wav")
                if not os.path.exists(audio_file_path):
                    audio_file_path = os.path.join("Data", "MP3 or WAV", "MP3", audio_file + ".mp3")
            else:
                audio_file_path = os.path.join("Data", "MP3 or WAV", "WAV" if audio_file_type == "wav" else "MP3", audio_file)
        else:
            audio_file_path = audio_file

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError
        
        native_sample_rate = librosa.get_samplerate(audio_file_path)
        amplitudes, sample_rate = librosa.load(audio_file_path, sr=min(native_sample_rate, target_sample_rate if target_sample_rate else float("inf")), mono=True)
        time = np.arange(len(amplitudes)) / sample_rate

        audio_df = pd.DataFrame({
            "Time": time,
            "Amplitude": amplitudes
        })

        return cls(audio_df)
    
    @classmethod
    def from_notes_list(cls, input_notes: list[Note | str], duration: float = float("inf"), sample_rate: float = 2000) -> AudioSignal:
        """
        Reads a list of Note objects of strs, and will create a AudioSignal with the Note data

        If there's a string in the inputted list, it'll be converted to a Note object with the default
        values for most of the attributes

        If the duration is by default set to float("inf"), which means the AudioSignal will be as long as the longest and last note,
        but if a value is given, the recording will be clipped to the inputted duration

        Args:
            input_notes (list[Note | str]): The list of notes to use for the AudioSignal
            duration (float): The duration of the recording in secs, by default inf, which makes the signal as long as necessary to contain all the notes
            sample_rate (float): The number of samples per sec in the audio signal, more means higher quality but more memory needed, less means lower quality but less memory needed
        
        Returns:
            AudioSignal: The AudioSignal object from the list of Notes and strs given
        """
        notes_to_use: list[Note] = []
        max_duration = 0
        if note in input_notes:
            if isinstance(note, str):
                notes_to_use.append(Note.from_string(note))
            elif isinstance(note, Note):
                notes_to_use.append(note)
            else:
                raise TypeError
            temp_duration = notes_to_use[-1].delay + notes_to_use[-1].duration
            if temp_duration > max_duration:
                max_duration = temp_duration

        if duration == float("inf"):
            duration = max_duration

        total_samples = int(sample_rate * duration)
        time = np.arange(total_samples) / sample_rate

        combined_amplitudes = np.zeros(total_samples)

        for note in set(notes_to_use):
            start_time = note.delay
            end_time = min(duration, note.delay + note.duration)

            if start_time >= duration:
                continue

            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)

            local_total_samples = end_idx - start_idx
            local_time = np.arange(local_total_samples) / sample_rate

            wave = note.amplitude * np.sin(2 * np.pi * note.get_freq() * local_time + note.phase)
            combined_amplitudes[start_idx:end_idx] += wave

        max_amp = np.max(np.abs(combined_amplitudes))
        if max_amp > 0:
            combined_amplitudes /= max_amp
        
        audio_df = pd.DataFrame({
            "Time": time,
            "Amplitude": combined_amplitudes
        })

        return cls(audio_df)
    
    @classmethod
    def from_ft_data(cls, ft_data: dict[str, np.ndarray | float | int | bool] | FourierTransformData, sample_rate: float = 2000, mode: int = 0) -> AudioSignal:
        """
        Reads the data from a Fourier Transform and reconstructs the original audio using the information

        Args:
            ft_data (dict[str, numpy.ndarray | float | int | bool] | FourierTransformData): The output from a Fourier Transform function
            sample_rate (float): The sample_rate to use if the Fourier Transform data doesn't contain it, will change the length and quality of the reconstructed audio
            mode (int): 0 is the default where a perfect reconstruction is attempted, 1 is a robotic mode where the phase information is removed, 2 is a crunchy mode where the phase information is randomized
        Returns:
            AudioSignal: The AudioSignal object from the Fourier Transform data given
        """
        if isinstance(ft_data, FourierTransformData):
            ft_data = ft_data.to_ft_dict()

        time_starts = ft_data["times"]
        freqs = ft_data["freqs"]
        amp_matrix = ft_data["amp_matrix"].T
        phase_matrix = ft_data["phase_matrix"].T
        true_sr = ft_data.get("sample_rate", sample_rate)

        NFFT = (len(freqs) - 1) * 2 + len(freqs) % 2
        noverlap = NFFT - round(true_sr * (time_starts[1] - time_starts[0]))

        true_NFFT = ft_data.get("NFFT", NFFT)
        true_noverlap = ft_data.get("noverlap", noverlap)
        
        step = true_NFFT - true_noverlap
        total_samples = (len(time_starts) - 1) * step + true_NFFT
        
        combined_amplitudes = np.zeros(total_samples)
        window_sum = np.zeros(total_samples)

        # --- THE WOLA FIX: Define the synthesis window ---
        use_hanning = ft_data.get("useHanning", False)
        synthesis_window = np.hanning(true_NFFT) if use_hanning else np.ones(true_NFFT)
        
        # We must square the window to perfectly normalize the overlap later!
        window_squared = synthesis_window ** 2

        for time_idx in range(len(time_starts)):
            current_amps = amp_matrix[time_idx]
            if mode == 0:
                current_phases = phase_matrix[time_idx]
            elif mode == 1:
                current_phases = np.zeros(len(current_amps))
            elif mode == 2:
                current_phases = np.random.uniform(-np.pi, np.pi, size=len(current_amps))
            local_time = np.arange(true_NFFT) / true_sr
            chunk_wave = np.zeros(true_NFFT)

            for freq_idx in range(len(freqs)):
                wave = current_amps[freq_idx] * np.cos(2 * np.pi * freqs[freq_idx] * local_time + current_phases[freq_idx])

                # if freq_idx != 0 and freq_idx != len(freqs) - 1:
                #     wave *= 2

                chunk_wave += wave
            
            chunk_wave /= true_NFFT

            chunk_wave *= synthesis_window
        
            start_idx = time_idx * step
            end_idx = start_idx + true_NFFT

            combined_amplitudes[start_idx:end_idx] += chunk_wave
            window_sum[start_idx:end_idx] += window_squared
        
        window_sum[window_sum == 0] = 1
        combined_amplitudes /= window_sum

        max_amp = np.max(np.abs(combined_amplitudes))
        if max_amp > 0:
            combined_amplitudes /= max_amp

        output_time = np.arange(total_samples) / true_sr
        
        audio_df = pd.DataFrame({
            "Time": output_time,
            "Amplitude": combined_amplitudes
        })

        return cls(audio_df)
    
    def to_audio_file(self, output_name: str, use_absolute_path: bool = False) -> None:
        """
        Exports the time vs amplitude in the AudioSignal as a .wav file in the directory 'Data/MP3 or WAV/WAV'

        Args:
            output_name (str): The name of the output file, doesn't need to have the suffix ".wav", as it would be added for you if it's not there
            use_absolute_path (bool): Whether to save to the Data directory, or to use a different absolute path
        """
        if not use_absolute_path:
            output_file_path = os.path.join("Data", "MP3 or WAV", "WAV", output_name if output_name.endswith(".wav") else output_name + ".wav")
        else:
            output_file_path = output_name

        amplitude = self.time_vs_amplitude["Amplitude"].astype(np.float32).values
        
        sample_rate = self.get_sample_rate()

        wavfile.write(output_file_path, int(sample_rate), amplitude)

    def to_csv_file(self, output_name: str, use_absolute_path: bool = False) -> None:
        """
        Exports the time vs amplitude in the AudioSignal as a .csv file in the directory 'Data/CSVs'

        Args:
            output_name (str): The name of the output file, doesn't need to have the suffix ".csv", as it would be added for you if it's not there
            use_absolute_path (bool): Whether to save to the Data directory, or to use a different absolute path
        """
        if not use_absolute_path:
            output_file_path = os.path.join("Data", "CSVs", output_name if output_name.endswith(".csv") else output_name + ".csv")
        else:
            output_file_path = output_name

        output_audio_df = self.time_vs_amplitude.rename(columns={"Time": "Time (s)"})
        output_audio_df.to_csv(output_file_path, index=False)

    def to_ft_data(self, ft_func: Callable[[pd.DataFrame | AudioSignal, int, int, bool], FourierTransformData]) -> FourierTransformData:
        """
        Runs the AudioSignal object through the given Fourier Transform function and outputs the data as
        a FourierTransformData object

        Args:
            ft_func (Callable[[pd.DataFrame | AudioSignal, int, int, bool], FourierTransformData]): The Fourier Transform function to do an analysis on AudioSignal
        
        Returns:
            FourierTransformData: The results output from the Fourier Transform function
        """
        return ft_func(self)

    def get_sample_rate(self) -> float:
        """
        Returns the sample rate of the saved time vs amplitude audio data

        Returns:
            float: The sample rate of the saved audio data
        """
        time = self.time_vs_amplitude["Time"].values

        if len(time) > 1:
            return int(round(1.0 / (time[1] - time[0])))
        
        return 44100
    
    def get_audio_duration(self) -> float:
        """
        Returns the duration of the saved time vs amplitude audio data

        Returns:
            float: The duration of the saved audio data
        """
        time = self.time_vs_amplitude["Time"].values
        return time[-1] - time[0]

    def pad_zeros(self, num_zeros: int) -> None:
        """
        Pads the end of the audio signal with num_zeros 0's

        This directly edits the time_vs_amplitude Pandas DataFrame stored in the object

        Args:
            num_zeros (int): The number of 0's to pad the end of the audio file with
        """
        sample_rate = self.get_sample_rate()
        amplitude = self.time_vs_amplitude["Amplitude"].astype(np.float32).values

        amps = np.pad(amplitude, (0, num_zeros))

        time_arr = np.arange(len(amplitude)) / sample_rate

        self.time_vs_amplitude = pd.DataFrame({
            "Time": time_arr,
            "Amplitude": amps
        })

    def apply_speed_change(self, speed_factor: float = 1) -> None:
        """
        Changes the speed of the saved audio data by speed_factor

        Args:
            speed_factor (float): The speed factor to multiply the saved times by
        """
        self.time_vs_amplitude["Time"] /= speed_factor
        self.time_vs_amplitude["Time"] -= self.time_vs_amplitude["Time"].min()
        self.time_vs_amplitude = self.time_vs_amplitude.sort_values(by="Time")

    def reverse_audio(self) -> None:
        """
        Reverses the saved audio data
        """
        self.apply_speed_change(-1)
        # self.time_vs_amplitude["Amplitude"] = self.time_vs_amplitude["Amplitude"].values[::-1]

    def play_audio_sound(self) -> None:
        """
        Takes the time vs amplitude data of the AudioSignal and plays it through the speakers
        """
        time = self.time_vs_amplitude["Time"].values
        amplitude = self.time_vs_amplitude["Amplitude"].values

        sample_rate = self.get_sample_rate()

        normalized_amplitude = amplitude / np.max(np.abs(amplitude))
        audio_data = normalized_amplitude.astype(np.float32)

        sd.play(audio_data, sample_rate)
        sd.wait()

class FourierTransformData:
    """
    A class to hold the output of the Fourier Transform functions and to manipulate the output easily

    Attributes:
        amp_matrix (numpy.ndarray): A 2D matrix of the amplitudes for different frequencies over time, X axis is time, Y axis is the frequency
        phase_matrix (numpy.ndarray): A 2D matrix of the phase for different frequencies over time, X axis is time, Y axis is the frequency
        freqs (numpy.ndarray): A 1D array of frequencies, these are the frequencies outputted by the Fourier Transform function
        times (numpy.ndarray):  A 1D array of times, these are the time intervals outputted by the Fourier Transform function
        sample_rate (float): The sample rate that the Fourier Transform used, this is stored to make reconstructing the audio easier
        NFFT (int): The NFFT that the Fourier Transform used, this is stored to make reconstructing the audio easier, but can be calculated from the audio data
        noverlap (int): The noverlap that the Fourier Transform used, this is stored to make reconstructing the audio easier, but can be calculated from the audio data
        useHanning (bool): Whether or not a Hanning window was used during the Fourier Transform
    """

    def __init__(self, amp_matrix: np.ndarray, phase_matrix: np.ndarray, freqs: np.ndarray, times: np.ndarray, sample_rate: float, NFFT: int, noverlap: int, useHanning: bool) -> None:
        """
        Initializes the FourierTransformData object

        Args:
            amp_matrix (numpy.ndarray): A 2D matrix of the amplitudes for different frequencies over time, X axis is time, Y axis is the frequency, this is taken from the Fourier Transform function output
            phase_matrix (numpy.ndarray): A 2D matrix of the phase for different frequencies over time, X axis is time, Y axis is the frequency, this is taken from the Fourier Transform function output
            freqs (numpy.ndarray): A 1D array of frequencies, these are the frequencies outputted by the Fourier Transform function, this is taken from the Fourier Transform function output
            times (numpy.ndarray):  A 1D array of times, these are the time intervals outputted by the Fourier Transform function, this is taken from the Fourier Transform function output
            sample_rate (float): The sample rate that the Fourier Transform used, this is stored to make reconstructing the audio easier, this is taken from the Fourier Transform function output
            NFFT (int): The NFFT that the Fourier Transform used, this is stored to make reconstructing the audio easier, this is taken from the Fourier Transform function output
            noverlap (int): The noverlap that the Fourier Transform used, this is stored to make reconstructing the audio easier, this is taken from the Fourier Transform function output
        """
        self._validate_inputs(amp_matrix, phase_matrix, freqs, times, sample_rate, NFFT, noverlap, useHanning)

        self.amp_matrix = amp_matrix
        self.phase_matrix = phase_matrix
        self.freqs = freqs
        self.times = times
        self.sample_rate = sample_rate
        self.NFFT = NFFT
        self.noverlap = noverlap
        self.useHanning = useHanning
    
    def __eq__(self, other: FourierTransformData) -> bool:
        return (np.array_equal(self.amp_matrix, other.amp_matrix)) and\
               (np.array_equal(self.phase_matrix, other.phase_matrix)) and\
               (np.array_equal(self.freqs, other.freqs)) and\
               (np.array_equal(self.times, other.times)) and\
               (self.sample_rate == other.sample_rate) and\
               (self.NFFT == other.NFFT) and\
               (self.noverlap == other.noverlap) and\
               (self.useHanning == other.useHanning)
    
    @classmethod
    def from_ft_dict(cls, ft_dict: dict[str, np.ndarray | float | int]) -> FourierTransformData:
        """
        Initialize a FourierTransformData object using a dictionary

        Args:
            ft_dict (dict[str, numpy.ndarray | float | int]): The dictionary that contains the output from the Fourier Transform function
        
        Returns:
            FourierTransformData: The FourierTransformData initialized from the inputted dictionary
        """
        return cls(
            ft_dict.get("amp_matrix"),
            ft_dict.get("phase_matrix"),
            ft_dict.get("freqs"),
            ft_dict.get("times"),
            ft_dict.get("sample_rate"),
            ft_dict.get("NFFT"),
            ft_dict.get("noverlap"),
            ft_dict.get("useHanning"),
        )

    @classmethod
    def from_csv_file(cls, file_name: str, use_absolute_path: bool = False) -> FourierTransformData:
        """
        Initialize a FourierTransformData object from a .csv file

        Args:
            file_name (str): The name of the .csv file, don't add any directories before it, and if the name doesn't end in ".csv", the program will automatically add it
            use_absolute_path (bool): Whether to load from the Data directory, or to use a different absolute path
        
        Returns:
            FourierTransformData: The FourierTransformData initialized from the inputted .csv file
        """
        if not use_absolute_path:
            if not file_name.endswith(".csv"):
                file_name = file_name + ".csv"
            file_path = os.path.join("Data", "CSVs", file_name)
        else:
            file_path = file_name

        with open(file_path, "r") as f:
            sample_rate = float(f.readline().strip().split(",")[1])
            NFFT = int(f.readline().strip().split(",")[1])
            noverlap = int(f.readline().strip().split(",")[1])
            useHanning = True if int(f.readline().strip().split(",")[1]) == 1 else False

            times = np.array(f.readline().strip().split(",")[1:], dtype=float)
            freqs = np.array(f.readline().strip().split(",")[1:], dtype=float)

            f.readline()
            amp_matrix = np.array([f.readline().strip().split(",") for _ in range(len(freqs))], dtype=float)

            f.readline()
            phase_matrix = np.array([f.readline().strip().split(",") for _ in range(len(freqs))], dtype=float)

            return cls(
                amp_matrix,
                phase_matrix,
                freqs,
                times,
                sample_rate,
                NFFT,
                noverlap,
                useHanning,
            )

    @classmethod
    def from_npz_file(cls, file_name: str, use_absolute_path: bool = False) -> FourierTransformData:
        """
        Initialize a FourierTransformData object from a .npz file

        Args:
            file_name (str): The name of the .npz file, don't add any directories before it, and if the name doesn't end in ".npz", the program will automatically add it
            use_absolute_path (bool): Whether to load from the Data directory, or to use a different absolute path
        
        Returns:
            FourierTransformData: The FourierTransformData initialized from the inputted .npz file
        """
        if not use_absolute_path:
            if not file_name.endswith(".npz"):
                file_name = file_name + ".npz"
            file_path = os.path.join("Data", "NPZs", file_name)
        else:
            file_path = file_name
        
        with np.load(file_path) as data:
            
            return cls(
                amp_matrix=data["amp_matrix"],
                phase_matrix=data["phase_matrix"],
                freqs=data["freqs"],
                times=data["times"],
                sample_rate=data["sample_rate"].item(),
                NFFT=data["NFFT"].item(),
                noverlap=data["noverlap"].item(),
                useHanning=data["useHanning"].item(),
            )

    @classmethod
    def _validate_inputs(cls, amp_matrix: np.ndarray, phase_matrix: np.ndarray, freqs: np.ndarray, times: np.ndarray, sample_rate: float, NFFT: int, noverlap: int, useHanning: bool) -> None:
        """
        Validates the inputs before initializing the FourierTransformData object

        Args:
            amp_matrix (numpy.ndarray): A 2D matrix of the amplitudes for different frequencies over time, X axis is time, Y axis is the frequency, this is taken from the Fourier Transform function output
            phase_matrix (numpy.ndarray): A 2D matrix of the phase for different frequencies over time, X axis is time, Y axis is the frequency, this is taken from the Fourier Transform function output
            freqs (numpy.ndarray): A 1D array of frequencies, these are the frequencies outputted by the Fourier Transform function, this is taken from the Fourier Transform function output
            times (numpy.ndarray):  A 1D array of times, these are the time intervals outputted by the Fourier Transform function, this is taken from the Fourier Transform function output
            sample_rate (float): The sample rate that the Fourier Transform used, this is stored to make reconstructing the audio easier, this is taken from the Fourier Transform function output
            NFFT (int): The NFFT that the Fourier Transform used, this is stored to make reconstructing the audio easier, this is taken from the Fourier Transform function output
            noverlap (int): The noverlap that the Fourier Transform used, this is stored to make reconstructing the audio easier, this is taken from the Fourier Transform function output
            useHanning (bool): Whether or not a Hanning window was used during the Fourier Transform
        
        Raises:
            ValueError: If any value of the args are None or NaN
            Exception: If the number of rows in amp_matrix doesn't equal the length of freqs
        """
        if amp_matrix is None or np.isnan(amp_matrix).any():
            raise ValueError
        
        if phase_matrix is None or np.isnan(phase_matrix).any():
            raise ValueError
        
        if freqs is None or np.isnan(freqs).any():
            raise ValueError
        
        if times is None or np.isnan(times).any():
            raise ValueError
        
        if sample_rate == None:
            raise ValueError
        
        if NFFT == None:
            raise ValueError
        
        if noverlap == None:
            raise ValueError
        
        if useHanning == None:
            raise ValueError
        
        if amp_matrix.shape[0] != len(freqs):
            raise Exception(f"amp_matrix has {amp_matrix.shape[0]} values for freqs, and there are {freqs.size} freqs")

    def apply_speed_change(self, speed_factor: float = 1) -> None:
        """
        Changes the speed of the Fourier Transform data by speed_factor

        Args:
            speed_factor (float): The speed factor to multiply the saved times by
        """
        self.sample_rate *= speed_factor
        self.freqs *= speed_factor
        self.times /= speed_factor

    def reverse_audio(self) -> None:
        """
        Reverses the Fourier Transform data
        """
        self.amp_matrix = self.reverse_matrix_across_time(self.amp_matrix)
        self.phase_matrix = self.reverse_matrix_across_time(self.phase_matrix)

    def reverse_matrix_across_time(self, matrix_in: np.ndarray) -> np.ndarray:
        """
        Reverses the input matrix across the time axis, such as the saved amplitude and phase matrices

        Args:
            matrix_in (numpy.ndarray): The inputted matrix
        
        Returns:
            numpy.ndarray: The outputted reversed matrix
        """
        transposed_matrix = matrix_in.T
        reversed_transposed_matrix = transposed_matrix[::-1]
        reversed_matrix = reversed_transposed_matrix.T
        return reversed_matrix

    def to_ft_dict(self) -> dict[str, np.ndarray | float | int]:
        """
        Returns the FourierTransformData object as a dictionary with the same attributes as key/value pairs
        
        Returns:
            dict[str, numpy.ndarray | float | int]: The data saved in the object converted to be stored in a dictionary instead
        """
        return {
            "amp_matrix": self.amp_matrix,
            "phase_matrix": self.phase_matrix,
            "freqs": self.freqs,
            "times": self.times,
            "sample_rate": self.sample_rate,
            "NFFT": self.NFFT,
            "noverlap": self.noverlap,
            "useHanning": self.useHanning,
        }
    
    def to_csv_file(self, output_name: str, use_absolute_path: bool = False) -> None:
        """
        Save the FourierTransformData object to a .csv file

        Args:
            output_name (str): The name of the .csv file, don't add any directories before it, and if the name doesn't end in ".csv", the program will automatically add it
            use_absolute_path (bool): Whether to save to the Data directory, or to use a different absolute path
        """
        if not use_absolute_path:
            if not output_name.endswith(".csv"):
                output_name = output_name + ".csv"
            output_file_path = os.path.join("Data", "CSVs", output_name)
        else:
            output_file_path = output_name
        
        
        with open(output_file_path, "w") as f:
            f.write(f"sample_rate,{self.sample_rate}\n")
            f.write(f"NFFT,{self.NFFT}\n")
            f.write(f"noverlap,{self.noverlap}\n")
            f.write(f"useHanning,{1 if self.useHanning else 0}\n")

            f.write("times," + ",".join(map(str, self.times)) + "\n")
            f.write("freqs," + ",".join(map(str, self.freqs)) + "\n")

            f.write("amp_matrix\n")
            np.savetxt(f, self.amp_matrix, delimiter=",")

            f.write("phase_matrix\n")
            np.savetxt(f, self.phase_matrix, delimiter=",")

    def to_npz_file(self, output_name: str, compress: bool = True, use_absolute_path: bool = False) -> None:
        """
        Save the FourierTransformData object to a .npz file

        Args:
            output_name (str): The name of the .npz file, don't add any directories before it, and if the name doesn't end in ".npz", the program will automatically add it
            use_absolute_path (bool): Whether to save to the Data directory, or to use a different absolute path
        """
        if not use_absolute_path:
            if not output_name.endswith(".npz"):
                output_name = output_name + ".npz"
            output_file_path = os.path.join("Data", "NPZs", output_name)
        else:
            output_file_path = output_name

        if compress:
            np.savez_compressed(
                output_file_path,
                amp_matrix = self.amp_matrix,
                phase_matrix = self.phase_matrix,
                freqs = self.freqs,
                times = self.times,
                sample_rate = np.array(self.sample_rate),
                NFFT = np.array(self.NFFT),
                noverlap = np.array(self.noverlap),
                useHanning = np.array(self.useHanning),
            )
        else:
            np.savez(
                output_file_path,
                amp_matrix = self.amp_matrix,
                phase_matrix = self.phase_matrix,
                freqs = self.freqs,
                times = self.times,
                sample_rate = np.array(self.sample_rate),
                NFFT = np.array(self.NFFT),
                noverlap = np.array(self.noverlap),
                useHanning = np.array(self.useHanning),
            )

    def to_AudioSignal(self) -> AudioSignal:
        """
        Converts the FourierTransformData into an AudioSignal object by inverting the Fourier Transform done to it
        
        Returns:
            AudioSignal: The reconstructed audio from the Fourier Transform data saved in the object
        """
        return AudioSignal.from_ft_data(self)


# 2 methods available
def custom_small_section_rdft(audio_slice_array: np.ndarray) -> dict[str, np.ndarray]:
    """
    Takes a small slice of a audio signal and applies a custom RDFT (Real Discreet Fourier Transform) on it

    Args:
        audio_slice_array (numpy.ndarray): The slice of the audio signal to apply a transform to, this is an array of amplitudes over time
    
    Returns:
        dict[str, numpy.ndarray]: A dictionary containing the amplitude and phase of each frequency bin with keys "amp" and "phase" respectively, the order of the amplitudes and phases in the array matches the order of the frequency array
    """
    # Double for loop
    # freq_to_amplitude_bins = []

    # for freq_bin_idx in range(len(audio_slice_array)//2 + 1):
    #     freq_to_amplitude_bins.append(0)

    #     for audio_data_idx in range(len(audio_slice_array)):
    #         audio_sample = audio_slice_array[audio_data_idx]
    #         if freq_bin_idx == 0 or audio_data_idx == 0:
    #             freq_to_amplitude_bins[freq_bin_idx] += audio_sample

    #         angle = (2 * np.pi * freq_bin_idx * audio_data_idx) / len(audio_slice_array)

    #         freq_to_amplitude_bins[freq_bin_idx] += audio_sample * (np.cos(angle) - 1j * np.sin(angle))
        
    # freq_to_amplitude_bins = np.array(freq_to_amplitude_bins)
    # freq_to_amplitude_bins *= 2
    # freq_to_amplitude_bins[0] /= 2
    # freq_to_amplitude_bins[-1] /= 2

    # return {
    #     "amp": np.abs(freq_to_amplitude_bins),
    #     "phase": np.angle(freq_to_amplitude_bins)
    # }

    # Matrix multiplication
    N = len(audio_slice_array)
    w_angle = 2 * np.pi / N
    w = np.cos(w_angle) - 1j * np.sin(w_angle)
    J, K = np.meshgrid(np.arange(N), np.arange(N//2 + 1))
    DFT_matrix = np.power(w, J * K)
    dft_data = DFT_matrix @ audio_slice_array

    dft_data *= 2
    dft_data[0] /= 2
    dft_data[-1] /= 2

    return {
        "amp": np.abs(dft_data),
        "phase": np.angle(dft_data)
    }

# 2 methods available
def custom_small_section_rfft(audio_slice_array: np.ndarray) -> dict[str, np.ndarray]:
    """
    Takes a small slice of a audio signal and applies a custom RFFT (Real Fast Fourier Transform) on it

    Args:
        audio_slice_array (numpy.ndarray): The slice of the audio signal to apply a transform to, this is an array of amplitudes over time
    
    Returns:
        dict[str, numpy.ndarray]: A dictionary containing the amplitude and phase of each frequency bin with keys "amp" and "phase" respectively, the order of the amplitudes and phases in the array matches the order of the frequency array
    """
    if not is_power_of_two(len(audio_slice_array)):
        zeros_needed = smallest_power_of_two_greater_than_n(len(audio_slice_array)) - len(audio_slice_array)
        audio_slice_array = np.pad(audio_slice_array, (0, zeros_needed), mode="constant")
    
    # # Slow - Unnecesary calculations
    # rfft_data = recursive_complex_fft(audio_slice_array)[:len(audio_slice_array)//2 + 1]
    
    # rfft_data *= 2
    # rfft_data[0] /= 2
    # rfft_data[-1] /= 2

    # return {
    #     "amp": np.abs(rfft_data),
    #     "phase": np.angle(rfft_data)
    # }

    # Fast - Cuts out the unnecessary calculations
    packed_array = audio_slice_array[0::2] + (1j * audio_slice_array[1::2])

    Z = recursive_complex_fft(packed_array)

    Z_rev = np.conj(np.concatenate(([Z[0]], Z[1:][::-1])))

    even_rfft = (Z + Z_rev) / 2
    odd_rfft = (Z - Z_rev) / 2j

    multipliers = get_twiddle_factors(len(audio_slice_array))

    rfft_data = np.empty(len(audio_slice_array) // 2 + 1, dtype=complex)

    rfft_data[:-1] = even_rfft + (multipliers * odd_rfft)
    rfft_data[-1] = even_rfft[0] - odd_rfft[0]
    
    rfft_data *= 2
    rfft_data[0] /= 2
    rfft_data[-1] /= 2

    return {
        "amp": np.abs(rfft_data),
        "phase": np.angle(rfft_data)
    }
    
def recursive_complex_fft(audio_slice_array: np.ndarray) -> np.ndarray:
    """
    Takes a small slice of a audio signal and applies a custom FFT (Fast Fourier Transform) on it

    Args:
        audio_slice_array (numpy.ndarray): The slice of the audio signal to apply a transform to, this is an array of amplitudes over time
    
    Returns:
        numpy.ndarray: An array containing complex numbers corresponding to another array of frequencies as the output of the FFT algorithm
    """
    if len(audio_slice_array) <= 1:
        return audio_slice_array
    
    even_indices = audio_slice_array[0::2]
    odd_indices = audio_slice_array[1::2]

    even_fft = recursive_complex_fft(even_indices)
    odd_fft = recursive_complex_fft(odd_indices)

    multipliers = get_twiddle_factors(len(audio_slice_array))

    first_half = even_fft + (multipliers * odd_fft)
    second_half = even_fft - (multipliers * odd_fft)

    return np.append(first_half, second_half)

def lib_small_section_rfft(audio_slice_array: np.ndarray) -> dict[str, np.ndarray]:
    """
    Takes a small slice of a audio signal and applies a library's RFFT (Real Fast Fourier Transform) on it

    Args:
        audio_slice_array (numpy.ndarray): The slice of the audio signal to apply a transform to, this is an array of amplitudes over time
    
    Returns:
        dict[str, numpy.ndarray]: A dictionary containing the amplitude and phase of each frequency bin with keys "amp" and "phase" respectively, the order of the amplitudes and phases in the array matches the order of the frequency array
    """
    rfft_data = np.fft.rfft(audio_slice_array)

    rfft_data *= 2
    rfft_data[0] /= 2
    rfft_data[-1] /= 2

    return {
        "amp": np.abs(rfft_data),
        "phase": np.angle(rfft_data)
    }

def apply_ft_on_chunks(audio_data: pd.DataFrame | AudioSignal, ft_func: Callable[[np.ndarray], dict[str, np.ndarray]], NFFT: int | float = 256, noverlap: int = 128, useHanning: bool = False) -> FourierTransformData:
    """
    Takes a small section Fourier Transform function, splits up the inputted audio signal, and applies the function to all the chunks
    Then accumilates all the amplitude and phase data into a 2D matrix to be outputted

    Args:
        audio_data (pandas.DataFrame | AudioSignal): The amplitude vs time data from an audio file
        ft_func (Callable[[numpy.ndarray], dict[str, numpy.ndarray]]): The function to use and apply to all the split up chunks
        NFFT (int | float): The number of data points in each chunk, if set to float("inf"), the entire audio signal will be used as 1 big chunk
        noverlap(int): The number of data points that overlap between adjacent chunks
        useHanning (bool): Whether or not to multiply each chunk by a Hanning window before applying the Fourier Transform function, best to use in random/non-periodic data, best not to use in periodic data
    
    Returns:
        FourierTransformData: The combined result of all the chunks that had a Fourier Transform function applied to them
    """
    if isinstance(audio_data, AudioSignal):
        audio_data = audio_data.time_vs_amplitude

    time = audio_data["Time"].values
    amplitude = audio_data["Amplitude"].values

    sample_rate = int(round(1.0 / (time[1] - time[0]))) if len(time) > 1 else 44100

    if NFFT == float("inf"):
        NFFT = len(amplitude)

    step = NFFT - noverlap

    num_chunks = len(range(0, len(amplitude) - NFFT + 1, step))
    num_freqs = NFFT // 2 + 1

    amp_matrix = np.zeros((num_chunks, num_freqs))
    phase_matrix = np.zeros((num_chunks, num_freqs))
    times = np.zeros(num_chunks)

    if useHanning:
        hanning_window = np.hanning(NFFT)

    for chunk_idx, i in enumerate(range(0, len(amplitude) - NFFT + 1, step)):
        chunk = amplitude[i : i + NFFT]
        if useHanning and len(chunk) == NFFT:
            ft_data = ft_func(chunk * hanning_window)
        elif useHanning and len(chunk) != NFFT:
            ft_data = ft_func(chunk * np.hanning(len(chunk)))
        else:
            ft_data = ft_func(chunk)
        amp_matrix[chunk_idx] = ft_data["amp"]
        phase_matrix[chunk_idx] = ft_data["phase"]
        times[chunk_idx] = (i + (NFFT / 2)) / sample_rate
    
    # This doesn't actually do any Fourier Transform math
    # Just creates a list of ints and scales it up for the output
    freqs = np.fft.rfftfreq(NFFT, 1/sample_rate)

    return FourierTransformData(
        amp_matrix.T,
        phase_matrix.T,
        freqs,
        times,
        sample_rate,
        NFFT,
        noverlap,
        useHanning=useHanning,
    )

def custom_rdft(audio_data: pd.DataFrame | AudioSignal, NFFT: int | float = 256, noverlap: int = 128, useHanning: bool = False) -> FourierTransformData:
    """
    Splits up the audio data into chunks, applies a Fourier Transform to each one, and puts all the data into a FourierTransformData object

    Args:
        audio_data (pandas.DataFrame | AudioSignal): The amplitude vs time data from an audio file
        NFFT (int | float): The number of data points in each chunk, if set to float("inf"), the entire audio signal will be used as 1 big chunk
        noverlap(int): The number of data points that overlap between adjacent chunks
        useHanning (bool): Whether or not to multiply each chunk by a Hanning window before applying the Fourier Transform function, best to use in random/non-periodic data, best not to use in periodic data
    
    Returns:
        FourierTransformData: The result of all the chunks that had a Fourier Transform function applied to them
    """
    return apply_ft_on_chunks(audio_data, custom_small_section_rdft, NFFT, noverlap, useHanning)

def custom_rfft(audio_data: pd.DataFrame | AudioSignal, NFFT: int | float = 256, noverlap: int = 128, useHanning: bool = False) -> FourierTransformData:
    """
    Splits up the audio data into chunks, applies a Fourier Transform to each one, and puts all the data into a FourierTransformData object

    Args:
        audio_data (pandas.DataFrame | AudioSignal): The amplitude vs time data from an audio file
        NFFT (int | float): The number of data points in each chunk, if set to float("inf"), the entire audio signal will be used as 1 big chunk
        noverlap(int): The number of data points that overlap between adjacent chunks
        useHanning (bool): Whether or not to multiply each chunk by a Hanning window before applying the Fourier Transform function, best to use in random/non-periodic data, best not to use in periodic data
    
    Returns:
        FourierTransformData: The result of all the chunks that had a Fourier Transform function applied to them
    """
    return apply_ft_on_chunks(audio_data, custom_small_section_rfft, NFFT, noverlap, useHanning)

def lib_rfft(audio_data: pd.DataFrame | AudioSignal, NFFT: int | float = 256, noverlap: int = 128, useHanning: bool = False) -> FourierTransformData:
    """
    Splits up the audio data into chunks, applies a Fourier Transform to each one, and puts all the data into a FourierTransformData object

    Args:
        audio_data (pandas.DataFrame | AudioSignal): The amplitude vs time data from an audio file
        NFFT (int | float): The number of data points in each chunk, if set to float("inf"), the entire audio signal will be used as 1 big chunk
        noverlap(int): The number of data points that overlap between adjacent chunks
        useHanning (bool): Whether or not to multiply each chunk by a Hanning window before applying the Fourier Transform function, best to use in random/non-periodic data, best not to use in periodic data
    
    Returns:
        FourierTransformData: The result of all the chunks that had a Fourier Transform function applied to them
    """
    return apply_ft_on_chunks(audio_data, lib_small_section_rfft, NFFT, noverlap, useHanning)

def calculate_true_amplitude_magnitude(ft_data: dict[str, np.ndarray | float | int] | FourierTransformData) -> np.ndarray:
    """
    Takes in the output from a Fourier Transform function, takes the amplitude matrix, and converts it into a matrix of true amplitude magnitude.
    This matrix can then be used in a spectrogram to plot the frequencies and their amplitudes over time

    Args:
        ft_data (dict[str, numpy.ndarray | float | int] | FourierTransformData): The output from a Fourier Transform function
    
    Returns:
        numpy.ndarray: The matrix of true amplitude magnitude
    """
    if isinstance(ft_data, FourierTransformData):
        ft_data = ft_data.to_ft_dict()

    return 20 * np.log10(ft_data["amp_matrix"]/ft_data["NFFT"] + 1e-10)

def calculate_decibel_intensity(ft_data: dict[str, np.ndarray | float | int] | FourierTransformData) -> np.ndarray:
    """
    Takes in the output from a Fourier Transform function, takes the amplitude matrix, and converts it into a matrix of decibel intensities.
    This matrix can then be used in a spectrogram to plot the frequencies and their amplitudes over time

    Args:
        ft_data (dict[str, numpy.ndarray | float | int] | FourierTransformData): The output from a Fourier Transform function
    
    Returns:
        numpy.ndarray: The matrix of decibel intensities
    """
    if isinstance(ft_data, FourierTransformData):
        ft_data = ft_data.to_ft_dict()

    power_matrix = ft_data["amp_matrix"] ** 2
    power_matrix[1:-1] /= 2
    if ft_data["useHanning"]:
        norm_factor = ft_data["sample_rate"] * np.sum(np.hanning(ft_data["NFFT"]) ** 2)
    else:
        norm_factor = ft_data["sample_rate"] * ft_data["NFFT"]
        
    return 10 * np.log10(power_matrix / norm_factor + 1e-20)

def add_note_lines(str_notes: list[str], horizontal: bool = False, min_freq: float | None = None, max_freq: float | None = None, min_octave: int = 0, max_octave: int = 8) -> None:
    """
    Adds lines and text in the current global plt graph to indicate where a note would be in the graph

    Args:
        str_notes (list[str]): A list of strs of length 3, with the first element being the note name (A, B, C, etc.), the second whether it's sharp, neutral, or flat (#, -, b), and the third being the octave number (0, 1, ..., 7, 8)\n
            The octave number can be X to loop through a bunch of octaves
        horizontal (bool): Whether or not to make the lines horizontal or vertical, useful for different graphs
        min_freq (float | None): By default is set to None to be disabled, but when given a note whose frequency is lower than this min_freq, then it won't be displayed
        max_freq (float | None): By default is set to None to be disabled, but when given a note whose frequency is higher than this max_freq, then it won't be displayed
        min_octave (int): The min octave number to have the "X" octave number loop through, inclusive
        max_octave (int): The max octave number to have the "X" octave number loop through, inclusive
    """
    class_notes: set[Note] = set()
    use_smaller_font = False

    for note in str_notes:
        if note[-1] == "X":
            for octave_num in range(min_octave, max_octave + 1):
                new_note = Note.from_string(note[:2] + str(octave_num))
                if min_freq and new_note.get_freq() < min_freq:
                    continue
                if max_freq and new_note.get_freq() > max_freq:
                    break
                class_notes.add(new_note)
                use_smaller_font = True
        else:
            new_note = Note.from_string(note)
            if (not min_freq or new_note.get_freq() >= min_freq) and (not max_freq or new_note.get_freq() <= max_freq):
                class_notes.add(new_note)

    freq_lines = {}

    for note in class_notes:
        freq_lines[str(note)] = note.get_freq()
    
    for note_name, freq in freq_lines.items():
        if horizontal:
            plt.axhline(y=freq, color="black", linestyle=":", alpha=0.7)
            plt.text(x=1.02, y=freq, s=note_name, color="black",
                     transform=plt.gca().get_yaxis_transform(),
                     ha="left", va="center", fontsize=5 if use_smaller_font else 10, rotation=0)
        else:
            plt.axvline(x=freq, color="black", linestyle=":", alpha=0.7)
            plt.text(x=freq, y=1.02, s=note_name, color="black",
                     transform=plt.gca().get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=5 if use_smaller_font else 10, rotation=90)
    
    plt.subplots_adjust(right=0.85)

def create_time_vs_amplitude_plots(audio_data: pd.DataFrame | AudioSignal, file_name: str, overwrite: bool = False, show_popup: bool = False, use_absolute_path: bool = False, width: int | None = None, use_grid: bool = True, only_points: bool = False) -> None:
    """
    Takes an AudioSignal object of Pandas DataFrame of amplitudes vs time and plots it
    This plot can be a popup or be saved as a .png

    Args:
        audio_data (pandas.DataFrame | AudioSignal): The audio data of an audio file containing the amplitudes over time
        file_name (str): The name of the .png file to export as, don't add any directories before it, and if the name doesn't end in ".png", the program will automatically add it
        overwrite (bool): Whether or not to overwrite the graph if it already exists
        show_popup (bool): Wether to show a popup and not save the graph, or not show a popup and save the graph
        use_absolute_path (bool): Whether to save to the Data directory, or to use a different absolute path
        width (int | None): Controls the width of the time vs amplitude graph, if set to None, it will get wider as the audio file gets longer, otherwise, it's set to width
        use_grid (bool): Whether or not to use a grid in the plot
        only_points (bool): Whether or not to only plot points in the graph
    """
    if not use_absolute_path:
        plot_dir = os.path.join("Graphs", "Time vs Amplitude", file_name if file_name.endswith(".png") else file_name + ".png")
    else:
        plot_dir = file_name

    if not overwrite and not show_popup and os.path.exists(plot_dir):
        return
    
    if isinstance(audio_data, AudioSignal):
        audio_data = audio_data.time_vs_amplitude

    fig_width = width if width is not None else (25 * len(audio_data["Time"]) / 1024)
    plt.figure(figsize=(fig_width, 6))

    if not only_points: plt.plot(audio_data["Time"], audio_data["Amplitude"], color="tab:blue", linewidth=2)
    else: plt.scatter(audio_data["Time"], audio_data["Amplitude"], color="tab:blue", s=0.5)

    plt.title("Time vs Amplitude audio graph", pad=50)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    if use_grid: plt.grid(True, alpha=0.3)
    plt.margins(x=0.01)

    if show_popup:
        plt.show()
    else:
        plt.savefig(plot_dir, bbox_inches="tight")

    plt.close()

def create_frequency_vs_amplitude_plots(audio_data: pd.DataFrame | AudioSignal | FourierTransformData, file_name: str, overwrite: bool = False, show_popup: bool = False,
                                        ft_func: Callable[[pd.DataFrame | AudioSignal, int, int, bool], FourierTransformData] = lib_rfft, NFFT: int | float = 256, noverlap: int = 128, useHanning: bool = False,
                                        notes_to_show: list[str] | None = None, min_octave: int = 0, max_octave: int = 8, use_absolute_path: bool = False) -> None:
    """
    Takes an AudioSignal object of Pandas DataFrame of amplitudes vs time and applies a Fourier Transform on it to get the amplitudes of a list of frequencies and then to plot them
    This plot can be a popup or be saved as a .png

    Args:
        audio_data (pandas.DataFrame | AudioSignal): The audio data of an audio file containing the amplitudes over time
        file_name (str): The name of the .png file to export as, don't add any directories before it, and if the name doesn't end in ".png", the program will automatically add it
        overwrite (bool): Whether or not to overwrite the graph if it already exists
        show_popup (bool): Wether to show a popup and not save the graph, or not show a popup and save the graph
        ft_func (Callable[[pandas.DataFrame | AudioSignal, int, int, bool], FourierTransformData]): The Fourier Transform function to use
        NFFT (int | float): The number of data points in each chunk, if set to float("inf"), the entire audio signal will be used as 1 big chunk
        noverlap(int): The number of data points that overlap between adjacent chunks
        useHanning (bool): Whether or not to multiply each chunk by a Hanning window before applying the Fourier Transform function, best to use in random/non-periodic data, best not to use in periodic data
        notes_to_show (list[str]): A list of strs of length 3, with the first element being the note name (A, B, C, etc.), the second whether it's sharp, neutral, or flat (#, -, b), and the third being the octave number (0, 1, ..., 7, 8)\n
            The octave number can be X to loop through a bunch of octaves
        min_octave (int): The min octave number to have the "X" octave number loop through, inclusive
        max_octave (int): The max octave number to have the "X" octave number loop through, inclusive
        use_absolute_path (bool): Whether to save to the Data directory, or to use a different absolute path
    """
    if not use_absolute_path:
        plot_dir = os.path.join("Graphs", "Freq vs Amplitude", file_name if file_name.endswith(".png") else file_name + ".png")
    else:
        plot_dir = file_name

    if not overwrite and not show_popup and os.path.exists(plot_dir):
        return
    
    if not isinstance(audio_data, FourierTransformData):
        if isinstance(audio_data, AudioSignal):
            audio_data = audio_data.time_vs_amplitude
        
        ft_data = ft_func(audio_data, NFFT, noverlap, useHanning).to_ft_dict()
    else:
        ft_data = audio_data

    plt.figure(figsize=(12, 6))

    for time_idx in range(len(ft_data["times"])):
        amplitudes = ft_data["amp_matrix"].T[time_idx]
        
        # amplitudes = np.array(amplitudes) * 2
        # amplitudes[0] /= 2
        # amplitudes[-1] /= 2
        if useHanning:
            amplitudes /= np.sum(np.hanning(NFFT))

        plt.plot(ft_data["freqs"], amplitudes, color="blue", alpha=0.3)

    plt.title("Frequency vs Amplitude graph", pad=50 if notes_to_show is not None else None)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")

    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.margins(x=0)

    if notes_to_show is not None:
        add_note_lines(notes_to_show, horizontal=False, min_freq=ft_data["freqs"].min(), max_freq=ft_data["freqs"].max(), min_octave=min_octave, max_octave=max_octave)

    if show_popup:
        plt.show()
    else:
        plt.savefig(plot_dir, bbox_inches="tight")

    plt.close()

def create_spectrogram_plots_lib(audio_data: pd.DataFrame | AudioSignal, file_name: str, overwrite: bool = False, show_popup: bool = False,
                                 NFFT: int = 256, noverlap: int = 128,
                                 notes_to_show: list[str] | None = None, min_octave: int = 0, max_octave: int = 8,
                                 color_map: str | Colormap = "viridis", use_absolute_path: bool = False) -> None:
    """
    Takes an AudioSignal object of Pandas DataFrame of amplitudes vs time and applies a Fourier Transform on it to get the amplitudes of a list of frequencies and then to plot them
    This plot can be a popup or be saved as a .png

    Args:
        audio_data (pandas.DataFrame | AudioSignal): The audio data of an audio file containing the amplitudes over time
        file_name (str): The name of the .png file to export as, don't add any directories before it, and if the name doesn't end in ".png", the program will automatically add it
        overwrite (bool): Whether or not to overwrite the graph if it already exists
        show_popup (bool): Wether to show a popup and not save the graph, or not show a popup and save the graph
        NFFT (int | float): The number of data points in each chunk, if set to float("inf"), the entire audio signal will be used as 1 big chunk
        noverlap(int): The number of data points that overlap between adjacent chunks
        notes_to_show (list[str]): A list of strs of length 3, with the first element being the note name (A, B, C, etc.), the second whether it's sharp, neutral, or flat (#, -, b), and the third being the octave number (0, 1, ..., 7, 8)\n
            The octave number can be X to loop through a bunch of octaves
        min_octave (int): The min octave number to have the "X" octave number loop through, inclusive
        max_octave (int): The max octave number to have the "X" octave number loop through, inclusive
        color_map (str | Colormap): Input an str to get a premade colormap, or input a custom one. Below are descriptions of what the color maps look like from low to high values\n
            "viridis" goes from dark purple, to teal, to bright yellow\n
            "neon_fire" goes from black, to dark purple, to red, to orange, to yellow, to white\n
            "classic_grayscale" goes from black to white\n
            "reverse_grayscale" goes from white to black\n
            "spectral_v1" goes from black, to blue, to cyan, to green, to yellow, to red\n
            "spectral_v2" goes from black, to dark purple, to light blue, to lime, to orange, to red\n
            "spectral_v3" goes from black, to dark purple, to light blue, to lime, to orange, to red, good for general purpose\n
            "spectral_v4" goes from black, to dark purple, to light blue, to lime, to orange, to red, good for seeing quiet details
        use_absolute_path (bool): Whether to save to the Data directory, or to use a different absolute path
    """
    if not use_absolute_path:
        plot_dir = os.path.join("Graphs", "Spectrograms", "lib", file_name if file_name.endswith(".png") else file_name + ".png")
    else:
        plot_dir = file_name

    if not overwrite and not show_popup and os.path.exists(plot_dir):
        return
    
    if isinstance(audio_data, AudioSignal):
        audio_data = audio_data.time_vs_amplitude

    time = audio_data["Time"].values
    amplitude = audio_data["Amplitude"].values

    duration = time[-1] - time[0]
    sample_rate = len(amplitude) // duration

    if isinstance(color_map, str):
        color_map = get_color_map(color_map)

    plt.figure(figsize=(10, 4))

    plt.specgram(audio_data["Amplitude"], Fs=sample_rate, cmap=color_map, NFFT=NFFT, noverlap=noverlap)

    plt.title("Spectrogram (Library FT)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Intensity", pad=0.15)

    plt.tight_layout()

    if notes_to_show is not None:
        freqs = np.fft.rfftfreq(NFFT, 1/sample_rate)
        add_note_lines(notes_to_show, horizontal=True, min_freq=freqs.min(), max_freq=freqs.max(), min_octave=min_octave, max_octave=max_octave)

    if show_popup:
        plt.show()
    else:
        plt.savefig(plot_dir, bbox_inches="tight")

    plt.close()

def create_spectrogram_plots_custom(audio_data: pd.DataFrame | AudioSignal | FourierTransformData, file_name: str, overwrite: bool = False, show_popup: bool = False,
                                    ft_func: Callable[[pd.DataFrame | AudioSignal, int, int, bool], FourierTransformData] = lib_rfft, NFFT: int = 256, noverlap: int = 128, useHanning: bool = False,
                                    notes_to_show: list[str] | None = None, min_octave: int = 0, max_octave: int = 8,
                                    use_intensity_db: bool = False, color_map: str | Colormap = "viridis", use_absolute_path: bool = False) -> None:
    """
    Takes an AudioSignal object of Pandas DataFrame of amplitudes vs time and applies a Fourier Transform on it to get the amplitudes of a list of frequencies and then to plot them
    This plot can be a popup or be saved as a .png

    Args:
        audio_data (pandas.DataFrame | AudioSignal | FourierTransformData): The audio data of an audio file containing the amplitudes over time
        file_name (str): The name of the .png file to export as, don't add any directories before it, and if the name doesn't end in ".png", the program will automatically add it
        overwrite (bool): Whether or not to overwrite the graph if it already exists
        show_popup (bool): Wether to show a popup and not save the graph, or not show a popup and save the graph
        ft_func (Callable[[pandas.DataFrame | AudioSignal, int, int, bool], FourierTransformData]): The Fourier Transform function to use
        NFFT (int | float): The number of data points in each chunk, if set to float("inf"), the entire audio signal will be used as 1 big chunk
        noverlap(int): The number of data points that overlap between adjacent chunks
        useHanning (bool): Whether or not to multiply each chunk by a Hanning window before applying the Fourier Transform function, best to use in random/non-periodic data, best not to use in periodic data
        notes_to_show (list[str]): A list of strs of length 3, with the first element being the note name (A, B, C, etc.), the second whether it's sharp, neutral, or flat (#, -, b), and the third being the octave number (0, 1, ..., 7, 8)\n
            The octave number can be X to loop through a bunch of octaves
        min_octave (int): The min octave number to have the "X" octave number loop through, inclusive
        max_octave (int): The max octave number to have the "X" octave number loop through, inclusive
        use_intensity_db (bool): Whether to plot the Fourier Transform data using True Amplitude Magnitude or to plot using Decibels
        color_map (str | Colormap): Input an str to get a premade colormap, or input a custom one. Below are descriptions of what the color maps look like from low to high values\n
            "viridis" goes from dark purple, to teal, to bright yellow\n
            "neon_fire" goes from black, to dark purple, to red, to orange, to yellow, to white\n
            "classic_grayscale" goes from black to white\n
            "reverse_grayscale" goes from white to black\n
            "spectral_v1" goes from black, to blue, to cyan, to green, to yellow, to red\n
            "spectral_v2" goes from black, to dark purple, to light blue, to lime, to orange, to red\n
            "spectral_v3" goes from black, to dark purple, to light blue, to lime, to orange, to red, good for general purpose\n
            "spectral_v4" goes from black, to dark purple, to light blue, to lime, to orange, to red, good for seeing quiet details
        use_absolute_path (bool): Whether to save to the Data directory, or to use a different absolute path
    """
    if not use_absolute_path:
        plot_dir = os.path.join("Graphs", "Spectrograms", "custom", file_name if file_name.endswith(".png") else file_name + ".png")
    else:
        plot_dir = file_name

    if not overwrite and not show_popup and os.path.exists(plot_dir):
        return
    
    if not isinstance(audio_data, FourierTransformData):
        if isinstance(audio_data, AudioSignal):
            audio_data = audio_data.time_vs_amplitude
        
        ft_data = ft_func(audio_data, NFFT, noverlap, useHanning).to_ft_dict()
    else:
        ft_data = audio_data.to_ft_dict()

    if isinstance(color_map, str):
        color_map = get_color_map(color_map)

    plt.figure(figsize=(10, 4))

    intensity_true_amplitude_magnitude = calculate_true_amplitude_magnitude(ft_data)
    intensity_db = calculate_decibel_intensity(ft_data)

    data_to_plot = intensity_true_amplitude_magnitude
    if use_intensity_db:
        data_to_plot = intensity_db

    # plt.pcolormesh(ft_data["times"], ft_data["freqs"], data_to_plot, shading="gouraud", cmap="viridis")
    # plt.pcolormesh(ft_data["times"], ft_data["freqs"], data_to_plot, shading="nearest", cmap="viridis")

    time_step = ft_data["times"][1] - ft_data["times"][0]
    freq_step = ft_data["freqs"][1] - ft_data["freqs"][0]

    extent_bounds = [
        ft_data["times"][0] - (time_step / 2),   # Left edge
        ft_data["times"][-1] + (time_step / 2),  # Right edge
        ft_data["freqs"][0] - (freq_step / 2),   # Bottom edge
        ft_data["freqs"][-1] + (freq_step / 2)   # Top edge
    ]

    plt.imshow(
        data_to_plot, 
        aspect="auto", 
        origin="lower", 
        extent=extent_bounds, 
        cmap=color_map,
        interpolation="bilinear",
        vmin=min(-120, np.min(data_to_plot)),
        vmax=np.max(data_to_plot)
    )

    plt.title("Spectrogram (Custom FT)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Intensity", pad=0.15)

    plt.tight_layout()

    if notes_to_show is not None:
        add_note_lines(notes_to_show, horizontal=True, min_freq=ft_data["freqs"].min(), max_freq=ft_data["freqs"].max(), min_octave=min_octave, max_octave=max_octave)

    if show_popup:
        plt.show()
    else:
        plt.savefig(plot_dir, bbox_inches="tight")

    plt.close()

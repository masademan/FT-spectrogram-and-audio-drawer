from fft_utils import (
    lib_rfft,
    custom_rfft,
    custom_rdft,
    AudioSignal,
    FourierTransformData,
    create_spectrogram_plots_lib,
    create_time_vs_amplitude_plots,
    create_spectrogram_plots_custom,
    create_frequency_vs_amplitude_plots,
)

def main() -> None:
    # audio_signal = AudioSignal.from_csv_file()
    # create_time_vs_amplitude_plots(audio_signal, "test", overwrite=True, show_popup=False)

    # create_spectrogram_plots_lib(audio_signal, "test", overwrite=True, show_popup=False)
    # create_spectrogram_plots_custom(audio_signal, "test", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=True, notes_to_show=["E-3", "G#3", "D-4", "G-4"])

    # create_frequency_vs_amplitude_plots(audio_signal, "test", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, notes_to_show=["E-3", "G#3", "D-4", "G-4"])

    # ft_data = lib_rfft(audio_signal)
    # ft_data.to_csv_file("test")

    # ft_data = FourierTransformData.from_npz_file("test")
    # audio_signal = ft_data.to_AudioSignal()
    # audio_signal.to_audio_file("test2.wav")

    # audio_signal = AudioSignal.from_audio_file("yippeee!!! daniel.mp3", target_sample_rate=None)
    # audio_signal.reverse_audio()
    # audio_signal.to_audio_file("yippeee!!! daniel - reversed.wav")

    # audio_signal = AudioSignal.from_audio_file("yippeee!!! daniel - reversed.wav", target_sample_rate=None)
    # audio_signal.reverse_audio()
    # audio_signal.to_audio_file("yippeee!!! daniel - reversed reversed.wav")

    audio_signal = AudioSignal.from_audio_file("yippeee!!! daniel.mp3", target_sample_rate=None)
    # audio_signal = AudioSignal.from_csv_file("secret_chord.csv")
    # create_time_vs_amplitude_plots(audio_signal, "yippeee!!! daniel", overwrite=True, show_popup=False, width=20, use_grid=False, only_points=True)
    # create_spectrogram_plots_custom(audio_signal, "test1", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False, color_map="spectral_v1")
    # create_spectrogram_plots_custom(audio_signal, "test2", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False, color_map="spectral_v2")
    # create_spectrogram_plots_custom(audio_signal, "test3", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False, color_map="spectral_v3")
    # create_spectrogram_plots_custom(audio_signal, "test4", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False, color_map="spectral_v4")
    # create_spectrogram_plots_custom(audio_signal, "test5", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False, color_map="neon_fire")
    # create_spectrogram_plots_custom(audio_signal, "test6", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False, color_map="classic_grayscale")
    # create_spectrogram_plots_custom(audio_signal, "test7", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False, color_map="reverse_grayscale")
    # create_spectrogram_plots_custom(audio_signal, "test8", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False, color_map="viridis")
    # create_time_vs_amplitude_plots(audio_signal, "test9")
    # audio_signal.apply_speed_change(0.5)
    # audio_signal.to_audio_file("yippeee!!! daniel - slowed down")
    # rfft_audio_signal = custom_rfft(audio_signal, useHanning=True)
    # rfft_audio_signal.apply_speed_change(0.5)
    # new_audio_signal = AudioSignal.from_ft_data(rfft_audio_signal, mode=0)
    # new_audio_signal_a = AudioSignal.from_ft_data(rfft_audio_signal, mode=0)
    # new_audio_signal_b = AudioSignal.from_ft_data(rfft_audio_signal, mode=1)
    # new_audio_signal_c = AudioSignal.from_ft_data(rfft_audio_signal, mode=2)
    # new_audio_signal_a.to_audio_file("normal")
    # new_audio_signal_b.to_audio_file("robotic")
    # new_audio_signal_c.to_audio_file("scratchy")
    # new_audio_signal.to_audio_file("test.wav")

    # create_spectrogram_plots_custom(audio_signal, "test1a", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=True)
    # create_spectrogram_plots_custom(audio_signal, "test2a", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False)
    # create_spectrogram_plots_lib(audio_signal, "test3a", overwrite=True, show_popup=False)

    # audio_signal = AudioSignal.from_audio_file("yippeee!!! daniel.mp3", target_sample_rate=3001)
    # create_spectrogram_plots_custom(audio_signal, "test1b", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=True)
    # create_spectrogram_plots_custom(audio_signal, "test2b", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_rfft, use_intensity_db=False)
    # create_spectrogram_plots_lib(audio_signal, "test3b", overwrite=True, show_popup=False)
    # new_audio_signal = AudioSignal.from_ft_data(lib_fft(audio_signal))
    # new_audio_signal.to_audio_file("test.wav")

    # create_spectrogram_plots_custom(audio_signal, "test", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_dft)
    # audio_signal.play_audio_sound()

    # create_frequency_vs_amplitude_plots(audio_signal, "test_custom", overwrite=True, useHanning=True, show_popup=False, ft_func=custom_dft)
    
    # audio_signal.to_audio_file("testA")
    # AudioSignal.from_ft_data(lib_fft(audio_signal)).to_audio_file("testB")
    # create_audio_with_dataframe(create_dataframe_with_notes(["E-3", "G#3", "D-4", "G-4"], 2, 2000), "test")
    # audio_signal.play_audio_sound()

    # All notes in an octave:
    # ["C-X", "C#X", "D-X", "D#X", "E-X", "F-X", "F#X", "G-X", "G#X", "A-X", "A#X", "B-X"]

if __name__ == "__main__":
    main()
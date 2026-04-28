import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.colors import LinearSegmentedColormap

def get_color_map(color_map_name: str) -> Colormap:
    colors = []

    # Neon fire
    if color_map_name == "neon_fire":
        colors = [
            "#000000",  # Absolute Silence (Black)
            "#1b0b2e",  # Very quiet (Deep Purple)
            "#5c005c",  # Quiet (Dark Magenta)
            "#c21533",  # Medium (Crimson Red)
            "#e87000",  # Loud (Orange)
            "#ffd500",  # Very Loud (Yellow)
            "#ffffff",  # Max Amplitude / 1.0 (White)
        ]

    # Classic grayscale
    if color_map_name == "classic_grayscale":
        colors = [
            "#000000",  # Silence (Black)
            "#ffffff",  # Maximum Amplitude (White)
        ]

    # Reversed classic grayscale
    if color_map_name == "reverse_grayscale":
        colors = [
            "#ffffff",  # Silence (White)
            "#000000",  # Maximum Amplitude (Black)
        ]

    # Spectral color map v1
    if color_map_name == "spectral_v1":
        colors = [
            "#000000",  # Silence (Black)
            "#0000ff",  # Very Quiet (Blue)
            "#00ffff",  # Quiet (Cyan)
            "#00ff00",  # Medium (Green)
            "#ffff00",  # Loud (Yellow)
            "#ff0000",  # Very Loud (Red)
        ]

    # Spectral color map v2
    if color_map_name == "spectral_v2":
        colors = [
            "#000000",  # Silence (Black)
            "#2c0a4d",  # Very Quiet (Dark purple)
            "#3d4dd6",  # Quiet (Light blue)
            "#69ff31",  # Medium (Lime)
            "#ffa100",  # Loud (Orange)
            "#ff0000",  # Very Loud (Red)
        ]

    # Spectral color map v3
    if color_map_name == "spectral_v3":
        colors = [
            (0.000, "#000000"),
            (0.297, "#2c0a4d"),
            (0.509, "#3d4dd6"),
            (0.676, "#69ff31"),
            (0.838, "#ffa100"),
            (1.000, "#ff0000"),
        ]

    # Spectral color map v4
    if color_map_name == "spectral_v4":
        colors = [
            (0.00, "#000000"),
            (0.05, "#2c0a4d"),
            (0.25, "#3d4dd6"),
            (0.50, "#69ff31"),
            (0.80, "#ffa100"),
            (1.00, "#ff0000"),
        ]
    
    if len(colors) > 0:
        # Tell Matplotlib to build a smooth gradient out of those colors!
        return LinearSegmentedColormap.from_list("spectrogram_color_map", colors, N=256)
    
    return plt.get_cmap("viridis")

def smallest_power_of_two_greater_than_n(n: int) -> int:
    """
    Gets the smallest power of 2 greater than n

    Args:
        n (int): The number that the power of 2 need to be greater than
    
    Returns:
        int: The smallest power of 2 greater than n
    """
    return 1 << (n - 1).bit_length()

def is_power_of_two(n: int) -> bool:
    """
    Checks if a number is a power of 2

    Args:
        n (int): The number to check
    
    Returns:
        bool: True if n is a power of 2, otherwise False
    """
    return n > 0 and (n & (n - 1)) == 0

def get_resourse_path(relative_path: str) -> str:
    """
    Converts a relative path to an absolute path, or a path for pyinstaller --onefile projects

    Args:
        relative_path (str): The relative path to make a path for any situation
    
    Returns:
        str: The converted path that'll work for normal cases and programs installed with pyinstaller
    """
    try:
        # PyInstaller creates a temp folder and stores path in sys._MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # If not running as a PyInstaller app, use the normal directory of this script
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

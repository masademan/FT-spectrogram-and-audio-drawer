import os
import time
import datetime
import numpy as np
import tkinter as tk
from math import ceil
from tkinter import ttk
import sounddevice as sd
from PIL import Image, ImageTk
from scipy.ndimage import label
from utils import get_color_map
import tkinter.filedialog as fd
import tkinter.messagebox as mb
from fft_utils import AudioSignal, FourierTransformData, lib_rfft

"""
FIX NOW:
make the frequency and time resolution independent of the height and width of the canvas

SpectrogramSynthesizer TODO:
allow for .png overlays
add control to how long the recording is, ie, allow for trimming or extending. and trimming doesn't actually delete the data, just stop showing part of the data in the UI and cuts the audio replay short, but once you export, the data actually gets trimmed
a button to clear the overlay
add a drop down and list for notes and note lines on the spectrogram
option to have 0 phase, random phase, semi-random phase, or saved phase (defaults to semi-random phase if saved phase doesn't exist)
option to see the saved phase matrix (defaults to random phase)
option to re-randomize the phase matrix
when viewing the phase matrix, have the ability to edit it with the drawing tools
the option to hide and show the overlay, and when the overlay is shown, make anything drawn red with the amplitude being reflected in the alpha value, and when the overlay is hidden, show the spectrogram using the normal color map
make the audio play in sections, like with a buffer and adding it to the cached data while the audio is playing, so that loops still can be cached
button to reverse the spectrogram canvas, and another button for the overlay (reversing across time)
quality control, from default to worse
make the audio synthesis and file uploading work with noverlap values that aren't NFFT // 2
allow for different layers, with a menu and checkboxs for enabling or disabling different layers
add a pause button to the audio playback, and when pausing it sets the starting point slider to the current pos

DrawingProgram TODO:
make the brush radius slider go from 0.5 to 2 with 0.5 resolution, and then from 2 to 50 with 1 resolution
brightness slider to make everything in the canvas brighter or dimmer, but only visual, and make the slider vertical. with a reset button to set the slider to default value
a home button to return to the original position and scaling
add zooming and panning in the canvas to more easily edit things
a button to clear the canvas
stack the tool buttons in 2 rows, one 3 tools long and the other 4 tools long
add borders, frames, and titles around the different controls
a layered pencil, similar to the non-additive, but even when letting go, it doesn't add to what it drew unless you click a button that says "new layer", ie, clearing the drawn_pixels set
an option to make the circle and square tool fill rather than just the outline
make the canvas have a scale, so the amount of data it can fit doesn't change, but the visual can get bigger or smaller
msg text in the far right bottom to display text instead of printing to console
write the coordinates of the mouse on the canvas in the bottom right corner of the window
make the wacom tablet work with the canvas better
make the grid align with the time and frequency axes
be able to select pixels and then shift them, along the time, frequency, or both axes
the option to see a plot of amplitude along the time axis or frequency axis, like in sebastian lague's vid (https://youtu.be/rRnOtKlg4jA?t=1733)

General TODO:
rename "fft_utils.py" to "signal_processing.py", and "gui_utils.py" to "synthesizer_daw.py"
ctrl + w to close the program
stylize the program, ie:
move the tools to the right part of the screen and the canvas toward the left part
add a color palette like a dark mode and light mode, and a button to switch between them or a drop down to select one
"""

DEBUG = True

COLOR_DESCRIPTIONS = {
    "spectral_v1": "Classic RGB: Black to Blue to Cyan to Green to Yellow to Red",
    "spectral_v2": "Alternative Spectral: Dark purple to Light blue to Lime",
    "spectral_v3": "Black to Purple to Blue to Lime to Red (General Purpose)",
    "spectral_v4": "Same as v3, but heavily tuned for seeing quiet details (Quiet Detailed Purpose)",
    "neon_fire": "Glowing FL Studio Style: Deep Purple to Crimson to White",
    "classic_grayscale": "Classic UI: Black (Silence) to White (Peak)",
    "reverse_grayscale": "Ink on Paper: White (Silence) to Black (Peak)",
    "viridis": "Default color map: Deep purple to Teal to Yellow (General Purpose)",
}

# HEAVILY AI assisted code
class AudioDrawingProgram:
    def __init__(self, root, width=800, height=400, name="Drawing Program"):
        self.root = root
        self.root.title(name)
        
        self.width = width
        self.height = height
        self.max_diagonal = int((width**2 + height**2)**0.5) + 1
        self.amp_matrix = np.zeros((self.height, self.width), dtype=float)
        
        # --- NEW WORKSPACE WITH AXES ---
        self.workspace_frame = tk.Frame(root)
        self.workspace_frame.pack(pady=10)
        
        # Y-Axis (Frequencies)
        self.y_axis_canvas = tk.Canvas(self.workspace_frame, width=80, height=self.height, bg=root.cget("bg"), highlightthickness=0)
        self.y_axis_canvas.grid(row=0, column=0, sticky="ns")
        
        # Main Drawing Board
        self.canvas = tk.Canvas(self.workspace_frame, width=self.width, height=self.height, bg="black", cursor="crosshair")
        self.canvas.grid(row=0, column=1)
        
        # X-Axis (Time)
        self.x_axis_canvas = tk.Canvas(self.workspace_frame, width=self.width, height=30, bg=root.cget("bg"), highlightthickness=0)
        self.x_axis_canvas.grid(row=1, column=1, sticky="ew")
        
        self.tk_image = None 
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        
        # --- PAN & ZOOM STATE VARIABLES ---
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_pan_x = None
        self.last_pan_y = None
        
        self.tk_image = None 
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        
        # --- NEW UI TOOLBAR ---
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=5)
        
        self.slider_frame = tk.Frame(self.btn_frame)
        self.slider_frame.pack(side=tk.LEFT, padx=15)
        
        # The independent label sits perfectly centered on top
        tk.Label(self.slider_frame, text="Opacity / Amplitude").pack(side=tk.TOP)
        
        # The bare slider sits underneath
        self.opacity_slider = tk.Scale(self.slider_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=200)
        self.opacity_slider.set(0.5) 
        self.opacity_slider.pack(side=tk.TOP)
        
        self.negate_opacity = tk.BooleanVar(value=False)
        tk.Checkbutton(self.slider_frame, text="Erase", variable=self.negate_opacity).pack(side=tk.TOP)

        # --- BRUSH CONTROLS ---
        self.brush_frame = tk.Frame(self.btn_frame)
        self.brush_frame.pack(side=tk.LEFT, padx=15)

        # --- (Add this inside __init__, under self.brush_slider.pack) ---
        self.show_brush_outline = tk.BooleanVar(value=True)
        tk.Checkbutton(self.brush_frame, text="Show Brush", variable=self.show_brush_outline, command=self.toggle_brush_outline).pack(side=tk.TOP)
        
        self.brush_outline_id = None # Tracks the hover circle!

        self.show_brush_while_drawing = tk.BooleanVar(value=False)
        tk.Checkbutton(self.brush_frame, text="Show Brush While Drawing", variable=self.show_brush_while_drawing).pack(side=tk.TOP)
        
        # Center the label just like the opacity slider!
        tk.Label(self.brush_frame, text="Brush Radius").pack(side=tk.TOP)
        
        self.brush_slider = tk.Scale(self.brush_frame, from_=0.5, to=50, resolution=0.5, orient=tk.HORIZONTAL, length=150)
        self.brush_slider.set(4) # Default back to your original size!
        self.brush_slider.pack(side=tk.TOP)

        # --- GRID CONTROLS ---
        self.grid_frame = tk.Frame(self.btn_frame)
        self.grid_frame.pack(side=tk.LEFT, padx=15)
        
        tk.Label(self.grid_frame, text="Grid X (Time)").pack(side=tk.TOP)
        self.grid_x_slider = tk.Scale(self.grid_frame, from_=1, to=100, orient=tk.HORIZONTAL, length=100)
        self.grid_x_slider.set(10)
        self.grid_x_slider.pack(side=tk.TOP)

        tk.Label(self.grid_frame, text="Grid Y (Freq)").pack(side=tk.TOP)
        self.grid_y_slider = tk.Scale(self.grid_frame, from_=1, to=100, orient=tk.HORIZONTAL, length=100)
        self.grid_y_slider.set(10)
        self.grid_y_slider.pack(side=tk.TOP)

        self.show_grid = tk.BooleanVar(value=False)
        tk.Checkbutton(self.grid_frame, text="Show Grid", variable=self.show_grid, command=self.update_grid).pack(side=tk.TOP)
        
        # Link the sliders to the grid update function!
        self.grid_x_slider.config(command=lambda _: self.update_grid())
        self.grid_y_slider.config(command=lambda _: self.update_grid())

        # --- VIEW CONTROLS ---
        self.view_frame = tk.Frame(self.btn_frame)
        self.view_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(self.view_frame, text="View").pack(side=tk.TOP)
        tk.Button(self.view_frame, text="🏠 Home", command=self.reset_view).pack(side=tk.TOP)

        # --- SHAPE TRACKING VARIABLES ---
        self.shape_anchor_x = None
        self.shape_anchor_y = None
        self.temp_shape_id = None # Tracks the temporary Tkinter visual outline

        # --- COLOR MAP CONTROLS ---
        self.cmap_frame = tk.Frame(self.btn_frame)
        self.cmap_frame.pack(side=tk.LEFT, padx=15)
        
        tk.Label(self.cmap_frame, text="Color Map").pack(side=tk.TOP)
        
        # 1. The Dropdown (Combobox)
        self.cmap_var = tk.StringVar(value="spectral_v3")
        self.cmap_dropdown = ttk.Combobox(self.cmap_frame, textvariable=self.cmap_var, 
                                          values=list(COLOR_DESCRIPTIONS.keys()), state="readonly", width=15)
        self.cmap_dropdown.pack(side=tk.TOP)
        
        # 2. The Dynamic Description Label
        self.cmap_desc_label = tk.Label(self.cmap_frame, text=COLOR_DESCRIPTIONS["spectral_v3"], 
                                        font=("Arial", 8, "italic"), fg="gray", wraplength=130, justify="center")
        self.cmap_desc_label.pack(side=tk.TOP)

        # --- UNDO / REDO CONTROLS ---
        self.history_frame = tk.Frame(self.btn_frame)
        self.history_frame.pack(side=tk.LEFT, padx=15)
        
        # Configure the max history limit!
        self.max_history = 25 
        self.undo_stack = []
        self.redo_stack = []
        
        tk.Label(self.history_frame, text="History").pack(side=tk.TOP)
        
        self.history_btn_frame = tk.Frame(self.history_frame)
        self.history_btn_frame.pack(side=tk.TOP)
        
        tk.Button(self.history_btn_frame, text="⬅", command=self.undo, width=4).pack(side=tk.LEFT, padx=2)
        tk.Button(self.history_btn_frame, text="➡", command=self.redo, width=4).pack(side=tk.LEFT, padx=2)

        # --- KEYBOARD SHORTCUTS ---
        # Note: We bind to self.root so they work no matter where your mouse is!
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Control-Z>", self.redo) # Tkinter sees Ctrl+Shift+z as capital Z!

        # --- PAN & ZOOM BINDINGS ---
        self.canvas.bind("<MouseWheel>", self.zoom_canvas)
        self.canvas.bind("<Button-4>", self.zoom_canvas) # Linux Scroll Up
        self.canvas.bind("<Button-5>", self.zoom_canvas) # Linux Scroll Down
        self.canvas.bind("<ButtonPress-3>", self.start_pan) # Right Click
        self.canvas.bind("<B3-Motion>", self.do_pan)
        
        # 2. The Tool Selector
        self.current_tool = tk.StringVar(value="pencil")
        tools = [("Pencil", "pencil"), ("Non-Additive", "non_add"), # ("Eraser", "eraser"),
                 ("Dropper", "dropper"), ("Square", "square"), ("Circle", "circle"), ("Line", "line"), ("Bucket", "bucket")]
        
        for text, val in tools:
            # indicatoron=0 makes radiobuttons look like normal push-buttons!
            tk.Radiobutton(self.btn_frame, text=text, variable=self.current_tool, value=val, indicatoron=0, width=10).pack(side=tk.LEFT, padx=2)
            
        # 3. Mouse Tracking Variables
        self.last_x = None
        self.last_y = None
        self.currently_clicking = False
        self.current_hover_color = None
        self.drawn_pixels = set() # Memory cache for the non-additive pencil!
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.cmap_dropdown.bind("<<ComboboxSelected>>", self.change_colormap)
        self.canvas.bind("<Motion>", self.hover) # Fires when moving WITHOUT clicking
        self.canvas.bind("<Leave>", self.hide_brush_outline) # Fires when mouse leaves the canvas

        self.change_colormap()

        self.render_matrix_to_image()

    def zoom_canvas(self, event):
        """Zooms in/out while keeping the mouse pinned to the exact same matrix coordinate."""
        # 1. Track exactly what matrix pixel the mouse is hovering over BEFORE zooming
        mx_before = self.pan_x + (event.x / self.zoom)
        my_before = self.pan_y + (event.y / self.zoom)
        
        # 2. Adjust Zoom
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.zoom = min(self.zoom * 1.2, 50.0) # Zoom In (Max 50x)
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.zoom = max(self.zoom / 1.2, 1.0)  # Zoom Out (Min 1x)
            
        # 3. Adjust pan so the mouse stays over the exact same matrix pixel AFTER zooming
        self.pan_x = mx_before - (event.x / self.zoom)
        self.pan_y = my_before - (event.y / self.zoom)
        
        self.hover(event)

        self.clamp_pan()
        self.render_matrix_to_image()
        
        if hasattr(self, 'time_slider'):
            self.draw_playhead(self.time_slider.get() * self.width / 100)

    def start_pan(self, event):
        self.last_pan_x = event.x
        self.last_pan_y = event.y

    def do_pan(self, event):
        if self.last_pan_x is None: return
        # Convert UI movement into Matrix movement (Inverse of zoom)
        self.pan_x -= (event.x - self.last_pan_x) / self.zoom
        self.pan_y -= (event.y - self.last_pan_y) / self.zoom

        self.hover(event)
        
        self.last_pan_x = event.x
        self.last_pan_y = event.y
        self.clamp_pan()
        self.render_matrix_to_image()
        
        if hasattr(self, 'time_slider'):
            self.draw_playhead(self.time_slider.get() * self.width / 100)

    def clamp_pan(self):
        """Prevents panning off the edge of the matrix."""
        max_pan_x = max(0.0, self.width - (self.width / self.zoom))
        max_pan_y = max(0.0, self.height - (self.height / self.zoom))
        self.pan_x = max(0.0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0.0, min(self.pan_y, max_pan_y))

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x, self.pan_y = 0.0, 0.0
        self.render_matrix_to_image()
        if hasattr(self, 'time_slider'):
            self.draw_playhead(self.time_slider.get() * self.width / 100)

    def draw_axes(self):
        self.y_axis_canvas.delete("all")
        self.x_axis_canvas.delete("all")
        
        # Fallbacks so DrawingProgram doesn't crash if run alone
        sr = getattr(self, 'canvas_sample_rate', 44100)
        duration = getattr(self, 'canvas_duration_seconds', 5.0)
        
        # Y-Axis (Freqs)
        for i in range(11):
            ui_y = i * (self.height / 10)
            my = self.pan_y + (ui_y / self.zoom) # Convert UI pos to Matrix pos
            freq = ((self.height - my) / self.height) * (sr / 2) # 0Hz is at the bottom!
            
            if 0 <= ui_y <= self.height:
                # --- SMART ANCHORS: Prevent Top/Bottom Cutoff ---
                if i == 0: anchor_pos = "ne"      # Top edge (Tuck inside)
                elif i == 10: anchor_pos = "se"   # Bottom edge (Tuck inside)
                else: anchor_pos = "e"            # Middle (Center normally)
                
                # Shifted to 75 to account for the new 80px width
                self.y_axis_canvas.create_text(75, ui_y, text=f"{int(freq)} Hz", anchor=anchor_pos, fill="gray", font=("Arial", 8))
                self.y_axis_canvas.create_line(75, ui_y, 80, ui_y, fill="gray")

        # X-Axis (Time)
        for i in range(11):
            ui_x = i * (self.width / 10)
            mx = self.pan_x + (ui_x / self.zoom)
            time_val = (mx / self.width) * duration
            
            if 0 <= ui_x <= self.width:
                # --- SMART ANCHORS: Prevent Left/Right Cutoff ---
                if i == 0: anchor_pos = "nw"      # Left edge (Tuck inside)
                elif i == 10: anchor_pos = "ne"   # Right edge (Tuck inside)
                else: anchor_pos = "n"            # Middle (Center normally)
                
                self.x_axis_canvas.create_text(ui_x, 15, text=f"{time_val:.2f}s", anchor=anchor_pos, fill="gray", font=("Arial", 8))
                self.x_axis_canvas.create_line(ui_x, 0, ui_x, 5, fill="gray")

    def save_state(self):
        """Saves a snapshot of the current matrix to the undo stack."""
        self.undo_stack.append(self.amp_matrix.copy())
        
        # Enforce the stack size limit so we don't blow out the RAM!
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0) 
            
        # The moment you draw something new, the redo "future" is destroyed
        self.redo_stack.clear() 

    def undo(self, event=None):
        if self.undo_stack:
            # Save the current state to the redo stack before going back
            self.redo_stack.append(self.amp_matrix.copy())
            
            # Pop the last saved state and make it the active matrix
            self.amp_matrix = self.undo_stack.pop()
            self.render_matrix_to_image()

    def redo(self, event=None):
        if self.redo_stack:
            # Save the current state to the undo stack before going forward
            self.undo_stack.append(self.amp_matrix.copy())
            
            # Pop the future state and make it the active matrix
            self.amp_matrix = self.redo_stack.pop()
            self.render_matrix_to_image()

    def change_colormap(self, event=None):
        """Updates the internal colormap, the UI description, and redraws the canvas!"""
        selected_cmap = self.cmap_var.get()
        
        # Update the mathematical colormap
        self.colormap = get_color_map(selected_cmap)
        
        # Update the UI description text
        self.cmap_desc_label.config(text=COLOR_DESCRIPTIONS[selected_cmap])
        
        # Instantly redraw the spectrogram with the new colors!
        self.render_matrix_to_image()

    def toggle_brush_outline(self):
        """Instantly hides the outline if the user unchecks the box."""
        if not self.show_brush_outline.get():
            self.hide_brush_outline()

    def hide_brush_outline(self, event=None):
        """Safely deletes the hover circle."""
        if self.brush_outline_id:
            self.canvas.delete(self.brush_outline_id)
            self.brush_outline_id = None
            self.current_hover_color = None

    def get_hover_color(self):
        """Returns the color name depending on the current state"""
        # Sets the outline to green if the mouse is clicked
        if self.currently_clicking:
            return "green"

        # If erase is on, make the color red, otherwise, light gray
        if self.negate_opacity.get():
            return "red"
        return "lightgray"

    def hover(self, event):
        if not self.show_brush_outline.get():
            return
            
        tool = self.current_tool.get()
        x, y = self.get_modified_coords(event, tool)
        r = self.brush_slider.get()
        if tool in ["dropper", "bucket"]: r = 4

        # Translate Matrix coords back to UI pixels to draw the circle
        ui_x = (x - self.pan_x) * self.zoom
        ui_y = (y - self.pan_y) * self.zoom
        ui_r = r * self.zoom # The brush visually scales up when you zoom in!

        if self.brush_outline_id and self.current_hover_color == self.get_hover_color():
            self.canvas.coords(self.brush_outline_id, ui_x - ui_r, ui_y - ui_r, ui_x + ui_r, ui_y + ui_r)
        else:
            color = self.get_hover_color()
            self.current_hover_color = color
            self.brush_outline_id = self.canvas.create_oval(
                ui_x - ui_r, ui_y - ui_r, ui_x + ui_r, ui_y + ui_r, outline=color, dash=(2, 2)
            )

    def start_stroke(self, event):
        self.save_state()
        self.hide_brush_outline()
        self.currently_clicking = True
        if self.show_brush_while_drawing.get():
            self.hover(event)

        tool = self.current_tool.get()
        x, y = self.get_modified_coords(event, tool)
        
        self.last_x, self.last_y = x, y
        self.shape_anchor_x, self.shape_anchor_y = x, y # Set the anchor for shapes!
        
        self.drawn_pixels = set()
        self.pixel_history = {} 
        self.stroke_distance = 0.0
        
        self.paint(event)

    def end_stroke(self, event):
        tool = self.current_tool.get()
        x, y = self.get_modified_coords(event, tool)

        brush_radius = self.brush_slider.get()
        
        temp_drawn_pixels = set()
        # --- BAKE THE SHAPE INTO THE MATRIX ---
        if tool in ["square", "circle", "line", "bucket"]:
            # 1. Erase the temporary Tkinter visual
            if self.temp_shape_id:
                self.canvas.delete(self.temp_shape_id)
                self.temp_shape_id = None
                
            # 2. Rasterize the Line into the NumPy Array!
            if tool == "line":
                intensity = self.opacity_slider.get() * (-1 if self.negate_opacity.get() else 1)
                
                distance = int(np.hypot(x - self.shape_anchor_x, y - self.shape_anchor_y))
                steps = max(1, distance)
                
                # We reuse our trusty interpolation logic to fill out the matrix!
                for i in range(steps + 1):
                    t = i / steps
                    interp_x = int(self.shape_anchor_x + (x - self.shape_anchor_x) * t)
                    interp_y = int(self.shape_anchor_y + (y - self.shape_anchor_y) * t)
                    
                    if 0 <= interp_x < self.width and 0 <= interp_y < self.height:
                        y_start = max(0, interp_y - brush_radius)
                        y_end = min(self.height, interp_y + brush_radius)
                        x_start = max(0, interp_x - brush_radius)
                        x_end = min(self.width, interp_x + brush_radius)

                        for py in range(y_start, y_end):
                            for px in range(x_start, x_end):
                                if (px - interp_x)**2 + (py - interp_y)**2 <= brush_radius**2:
                                    if (px, py) not in temp_drawn_pixels:
                                        temp_drawn_pixels.add((px, py))
                                        self.amp_matrix[py, px] += intensity
            
            elif tool == "square":
                intensity = self.opacity_slider.get() * (-1 if self.negate_opacity.get() else 1)
                T = brush_radius * 2  # The thickness of the square's outline
                
                # Figure out the boundaries (sorted handles dragging the mouse backwards!)
                x1, x2 = sorted([self.shape_anchor_x, x])
                y1, y2 = sorted([self.shape_anchor_y, y])
                
                # Constrain coordinates to the canvas size so we don't crash NumPy
                x1 = max(0, x1 - T//2)
                x2 = min(self.width, x2 + T//2)
                y1 = max(0, y1 - T//2)
                y2 = min(self.height, y2 + T//2)

                # Calculate the inner boundaries to prevent overlapping corners!
                y_inner_top = min(y1 + T, y2)
                y_inner_bottom = max(y1 + T, y2 - T)
                x_inner_left = min(x1 + T, x2)
                x_inner_right = max(x1 + T, x2 - T)
                
                # 1. Top Wall
                self.amp_matrix[y1:y_inner_top, x1:x2] += intensity
                # 2. Bottom Wall
                self.amp_matrix[y_inner_bottom:y2, x1:x2] += intensity
                # 3. Left Wall (sliced to sit perfectly between the top and bottom walls)
                self.amp_matrix[y_inner_top:y_inner_bottom, x1:x_inner_left] += intensity
                # 4. Right Wall (sliced to sit perfectly between the top and bottom walls)
                self.amp_matrix[y_inner_top:y_inner_bottom, x_inner_right:x2] += intensity

            elif tool == "bucket":
                intensity = self.opacity_slider.get() if not self.negate_opacity.get() else 0
                
                # Clamp coordinates to ensure we don't click out of bounds
                safe_x = max(0, min(self.width - 1, x))
                safe_y = max(0, min(self.height - 1, y))
                
                # 1. Get the exact amplitude of the pixel we clicked on
                target_amp = self.amp_matrix[safe_y, safe_x]
                
                # 2. Create a True/False mask of EVERY pixel on the board that matches this amplitude.
                # (We use < 1e-4 instead of '==' to protect against floating point rounding errors!)
                matching_pixels_mask = np.abs(self.amp_matrix - target_amp) < 1e-4
                
                # 3. SciPy Magic: Group all contiguous touching pixels into numbered blobs
                labeled_matrix, _ = label(matching_pixels_mask)
                
                # 4. Get the specific ID number of the blob we clicked on
                clicked_blob_id = labeled_matrix[safe_y, safe_x]
                
                # 5. Instantly inject the amplitude into just that specific blob!
                self.amp_matrix[labeled_matrix == clicked_blob_id] = intensity

            elif tool == "circle":
                intensity = self.opacity_slider.get()
                T = brush_radius # We expand outward by T and inward by T
                
                x1, x2 = sorted([self.shape_anchor_x, x])
                y1, y2 = sorted([self.shape_anchor_y, y])
                
                # Calculate the center point and the X/Y radii
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                rx = max(0.1, (x2 - x1) / 2.0)
                ry = max(0.1, (y2 - y1) / 2.0)
                
                # Define a safe bounding box that includes our brush thickness
                box_x1 = max(0, int(cx - rx - T - 1))
                box_x2 = min(self.width, int(cx + rx + T + 1))
                box_y1 = max(0, int(cy - ry - T - 1))
                box_y2 = min(self.height, int(cy + ry + T + 1))
                
                if box_x2 > box_x1 and box_y2 > box_y1:
                    # Generate a high-speed NumPy coordinate grid for this box
                    Y, X = np.ogrid[box_y1:box_y2, box_x1:box_x2]
                    
                    # Mathematical formula for the outer boundary of the ellipse
                    outer_mask = ((X - cx) / (rx + T))**2 + ((Y - cy) / (ry + T))**2 <= 1.0
                    
                    # Mathematical formula for the inner boundary (protect against negative radii!)
                    inner_rx = max(0.0001, rx - T)
                    inner_ry = max(0.0001, ry - T)
                    inner_mask = ((X - cx) / inner_rx)**2 + ((Y - cy) / inner_ry)**2 <= 1.0
                    
                    # The circle is the pixels inside the outer bounds, but outside the inner bounds
                    ring_mask = outer_mask & ~inner_mask
                    
                    # Slice the matrix and inject the opacity!
                    self.amp_matrix[box_y1:box_y2, box_x1:box_x2][ring_mask] += intensity

            # Re-render the matrix to show the baked shape
            self.amp_matrix = np.clip(self.amp_matrix, 0.0, 1.0)
            self.render_matrix_to_image()
        

        # Clean up variables for the next stroke
        self.last_x, self.last_y = None, None
        self.currently_clicking = False
        self.shape_anchor_x, self.shape_anchor_y = None, None
        self.drawn_pixels.clear() 
        self.pixel_history.clear()

        self.hide_brush_outline()
        self.hover(event)

    def get_modified_coords(self, event, tool):
        # 1. Transform UI Pixels into Matrix Indices!
        matrix_x = self.pan_x + (event.x / self.zoom)
        matrix_y = self.pan_y + (event.y / self.zoom)
        
        # Override the variables so the rest of the math operates directly on the Matrix!
        x, y = matrix_x, matrix_y
        
        # 2. CTRL: Snap to Grid
        if event.state & 0x0004:
            grid_x = self.grid_x_slider.get()
            grid_y = self.grid_y_slider.get()
            x = round(x / grid_x) * grid_x
            y = round(y / grid_y) * grid_y

        # 3. SHIFT: Orthogonal Lock
        if (event.state & 0x0001) and self.shape_anchor_x is not None:
            if tool == "line":
                if abs(x - self.shape_anchor_x) > abs(y - self.shape_anchor_y):
                    y = self.shape_anchor_y 
                else:
                    x = self.shape_anchor_x 
            elif tool in ["square", "circle"]:
                dx, dy = x - self.shape_anchor_x, y - self.shape_anchor_y
                size = max(abs(dx), abs(dy))
                x = self.shape_anchor_x + (size * (1 if dx > 0 else -1))
                y = self.shape_anchor_y + (size * (1 if dy > 0 else -1))

        return int(x), int(y)

    def update_grid(self):
        self.canvas.delete("grid_line")
        if self.show_grid.get():
            step_x = self.grid_x_slider.get()
            step_y = self.grid_y_slider.get()
            
            start_x = int(self.pan_x // step_x) * step_x
            for mx in range(start_x, int(self.pan_x + self.width / self.zoom) + step_x, step_x):
                ui_x = (mx - self.pan_x) * self.zoom # Translate matrix line back to UI screen!
                if 0 <= ui_x <= self.width:
                    self.canvas.create_line(ui_x, 0, ui_x, self.height, fill="gray", dash=(2, 4), tags="grid_line")
                    
            start_y = int(self.pan_y // step_y) * step_y
            for my in range(start_y, int(self.pan_y + self.height / self.zoom) + step_y, step_y):
                ui_y = (my - self.pan_y) * self.zoom
                if 0 <= ui_y <= self.height:
                    self.canvas.create_line(0, ui_y, self.width, ui_y, fill="gray", dash=(2, 4), tags="grid_line")

    def paint(self, event):
        x, y = event.x, event.y
        tool = self.current_tool.get()
        opacity = self.opacity_slider.get()

        if self.show_brush_while_drawing.get():
            self.hover(event)
        
        # Get the modified coordinates!
        x, y = self.get_modified_coords(event, tool)

        brush_radius = self.brush_slider.get()

        # --- SHAPE RENDERING (Visual Only) ---
        if tool in ["square", "circle", "line", "bucket", "dropper"]:
            # (Notice we removed the block that deletes the shape!)
            
            if tool == "line":
                if self.temp_shape_id:
                    # If it exists, just stretch it to the new mouse coordinates!
                    self.canvas.coords(self.temp_shape_id, self.shape_anchor_x, self.shape_anchor_y, x, y)
                else:
                    # If it doesn't exist yet, create it!
                    self.temp_shape_id = self.canvas.create_line(
                        self.shape_anchor_x, self.shape_anchor_y, x, y, 
                        fill="red", width=4, dash=(4, 4)
                    )
                    
            elif tool == "square":
                if self.temp_shape_id:
                    self.canvas.coords(self.temp_shape_id, self.shape_anchor_x, self.shape_anchor_y, x, y)
                else:
                    self.temp_shape_id = self.canvas.create_rectangle(
                        self.shape_anchor_x, self.shape_anchor_y, x, y, 
                        outline="red", width=4, dash=(4, 4)
                    )
                    
            elif tool == "circle":
                if self.temp_shape_id:
                    self.canvas.coords(self.temp_shape_id, self.shape_anchor_x, self.shape_anchor_y, x, y)
                else:
                    self.temp_shape_id = self.canvas.create_oval(
                        self.shape_anchor_x, self.shape_anchor_y, x, y, 
                        outline="red", width=4, dash=(4, 4)
                    )
                    
            elif tool == "dropper":
                safe_x = max(0, min(self.width - 1, int(x)))
                safe_y = max(0, min(self.height - 1, int(y)))
                sampled_amplitude = self.amp_matrix[safe_y, safe_x]
                self.opacity_slider.set(sampled_amplitude)
                
            return # Exit the function early!

        if self.last_x is None:
            self.last_x, self.last_y = x, y
        
        # If it's an eraser, we just subtract the opacity instead of adding it
        # intensity = -opacity if tool == "eraser" else opacity
        intensity = -opacity if self.negate_opacity.get() else opacity
        
        distance = int(np.hypot(x - self.last_x, y - self.last_y))
        steps = max(1, distance) 

        for i in range(1, steps + 1):
            t = i / steps
            interp_x = int(self.last_x + (x - self.last_x) * t)
            interp_y = int(self.last_y + (y - self.last_y) * t)
            
            # Increment the physical distance the brush has traveled
            self.stroke_distance += (distance / steps)
            
            # Apply the brush
            if 0 <= interp_x < self.width and 0 <= interp_y < self.height:
                y_start = max(0, interp_y - brush_radius)
                y_end = min(self.height, interp_y + brush_radius)
                x_start = max(0, interp_x - brush_radius)
                x_end = min(self.width, interp_x + brush_radius)
                
                for py in range(ceil(y_start), ceil(y_end)):
                    for px in range(ceil(x_start), ceil(x_end)):
                        
                        if (px - interp_x)**2 + (py - interp_y)**2 <= brush_radius**2:
                            if tool == "non_add":
                                if (px, py) not in self.drawn_pixels:
                                    self.amp_matrix[py, px] += intensity
                                    self.drawn_pixels.add((px, py))
                                    
                            else: # The Normal Pencil and Eraser!
                                # Find out what distance the brush was at when it last painted this pixel
                                last_painted_dist = self.pixel_history.get((px, py), -self.max_diagonal)
                                
                                # Only repaint if the brush has traveled completely away and returned!
                                if self.stroke_distance - last_painted_dist > (brush_radius * 4):
                                    self.amp_matrix[py, px] += intensity
                                    self.pixel_history[(px, py)] = self.stroke_distance

        self.last_x, self.last_y = x, y
        self.amp_matrix = np.clip(self.amp_matrix, 0.0, 1.0)
        self.render_matrix_to_image()

    def render_matrix_to_image(self):
        """The Render Engine: Crops the matrix based on zoom, and upscales it to the canvas."""
        
        # 1. Calculate Crop Bounds
        x1, y1 = int(self.pan_x), int(self.pan_y)
        x2 = int(self.pan_x + self.width / self.zoom)
        y2 = int(self.pan_y + self.height / self.zoom)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, max(x1+1, x2)), min(self.height, max(y1+1, y2))
        
        # 2. Extract visible slice and apply colors
        cropped_matrix = self.amp_matrix[y1:y2, x1:x2]
        colored_cropped = self.colormap(cropped_matrix)
        cropped_int = (colored_cropped[:, :, :3] * 255).astype(np.uint8)
        
        # 3. Scale the cropped slice UP to fill the UI
        display_pil = Image.fromarray(cropped_int).resize((self.width, self.height), Image.Resampling.NEAREST)
        
        # 4. Generate the FULL HD image exclusively for the "Save as PNG" feature!
        colored_full = self.colormap(self.amp_matrix)
        self.pil_image = Image.fromarray((colored_full[:, :, :3] * 255).astype(np.uint8))
        
        self.tk_image = ImageTk.PhotoImage(image=display_pil)
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)

        self.update_grid()
        self.draw_axes()

class SpectrogramSynthesizer(AudioDrawingProgram):
    def __init__(self, root, width=800, height=400, name="Spectrogram DAW"):
        # Initialize the base drawing program!
        super().__init__(root, width, height, name=name)

        self.canvas_sample_rate = 44100
        self.canvas_duration_seconds = 5.0
        
        # --- DAW TOOLBAR ---
        self.daw_frame = tk.Frame(root)
        self.daw_frame.pack(pady=5, fill=tk.X, padx=10)
        
        # 1. FILE I/O & OVERLAY FRAME
        self.io_frame = tk.LabelFrame(self.daw_frame, text="File & Overlay")
        self.io_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        tk.Button(self.io_frame, text="📂 Open (Ctrl+O)", command=self.open_file).pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        tk.Button(self.io_frame, text="💾 Save (Ctrl+S)", command=self.save_file).pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        self.root.bind("<Control-o>", self.open_file)
        self.root.bind("<Control-s>", self.save_file)
        
        self.import_mode = tk.StringVar(value="edit")
        tk.Radiobutton(self.io_frame, text="Import to Edit", variable=self.import_mode, value="edit", command=self.render_matrix_to_image).pack(side=tk.TOP, anchor=tk.W)
        tk.Radiobutton(self.io_frame, text="Import as Overlay", variable=self.import_mode, value="overlay", command=self.render_matrix_to_image).pack(side=tk.TOP, anchor=tk.W)
        
        self.overlay_opacity = tk.Scale(self.io_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, label="Overlay Opacity", command=lambda _: self.render_matrix_to_image())
        self.overlay_opacity.set(0.4)
        self.overlay_opacity.pack(side=tk.TOP, padx=5)
        
        # Overlay State
        self.overlay_matrix = np.zeros((self.height, self.width), dtype=float)
        self.has_overlay = False

        self.zero_phase_matrix = None
        self.random_phase_matrix = None
        self.semi_random_phase_matrix = None

        # 2. PLAYBACK ENGINE FRAME
        self.playback_frame = tk.LabelFrame(self.daw_frame, text="Playback Engine")
        self.playback_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        btn_frame = tk.Frame(self.playback_frame)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        self.play_btn = tk.Button(btn_frame, text="▶ Play", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=8, command=self.play_audio)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop", bg="#f44336", fg="white", font=("Arial", 10, "bold"), width=8, command=self.stop_audio)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        self.is_playing = False
        self.playback_start_time = 0.0
        self.playback_start_percent = 0.0
        self.total_playback_duration = 0.0

        self.undo_redo_counter = 0
        
        self.loop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_frame, text="Loop", variable=self.loop_var).pack(side=tk.LEFT, padx=5)
        
        self.time_slider = tk.Scale(self.playback_frame, from_=0.0, to=100.0, resolution=0.1, orient=tk.HORIZONTAL, label="Start Position (%)", length=300, command=lambda _: self.draw_playhead(self.time_slider.get() * self.width / 100))
        self.time_slider.pack(side=tk.TOP, padx=5)
        
        self.show_playhead_line = tk.BooleanVar(value=True)
        tk.Checkbutton(btn_frame, text="Show Playhead Line", variable=self.show_playhead_line, command=lambda: self.draw_playhead(self.time_slider.get() * self.width / 100)).pack(side=tk.LEFT, padx=5)

        # 3. SETTINGS FRAME
        self.settings_frame = tk.LabelFrame(self.daw_frame, text="Settings")
        self.settings_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        self.volume_slider = tk.Scale(self.settings_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, label="Volume")
        self.volume_slider.set(0.8)
        self.volume_slider.pack(side=tk.TOP, padx=5)
        
        self.speed_slider = tk.Scale(self.settings_frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label="Speed")
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side=tk.TOP, padx=5)
        
        # PLAYHEAD VISUALS
        self.playhead_triangle_id = None
        self.playhead_line_id = None
        self.draw_playhead(0) # Initialize at 0

    def play_audio(self, from_loop = False):
        """Compiles the canvas to audio and starts non-blocking playback."""
        if self.is_playing and not from_loop:
            self.stop_audio() # Stop any currently playing audio first
        
        if not hasattr(self, 'cached_base_audio') or self.cached_base_audio is None or self.undo_redo_counter != 0:
            if DEBUG: print("Synthesizing audio (Heavy Math)...")
            ft_data = self.generate_ft_data()
            signal = ft_data.to_AudioSignal()
            # Save the raw numpy array to the cache!
            self.cached_base_audio = signal.time_vs_amplitude["Amplitude"].astype(np.float32).values
            self.undo_redo_counter = 0
        else:
            if DEBUG: print("Using cached audio for seamless loop!")
        
        # 1. Apply Volume
        volume = self.volume_slider.get()
        audio_array = self.cached_base_audio * volume
        
        # 2. Apply Start Position
        start_percent = self.time_slider.get() / 100.0
        start_index = int(len(audio_array) * start_percent)
        audio_to_play = audio_array[start_index:]
        
        # 3. Apply Speed (By altering the sample rate, we change speed & pitch simultaneously)
        base_sample_rate = self.canvas_sample_rate
        speed = self.speed_slider.get()
        playback_sample_rate = int(base_sample_rate * speed)

        # 1. Smoothly fade out the last 500 samples to prevent a "pop"
        ideal_fade_len = int(playback_sample_rate * 0.02)

        fade_len = min(ideal_fade_len, len(audio_to_play) // 10)
        if fade_len > 0:
            audio_to_play[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            audio_to_play[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            
        # 2. Glue 0.2 seconds of pure zeros (silence) to the end of the track
        silence_padding = np.zeros(int(playback_sample_rate * 0.2), dtype=np.float32)
        audio_to_play = np.concatenate((audio_to_play, silence_padding))
        
        # 4. Save state for the visual playhead tracker
        self.is_playing = True
        self.playback_start_percent = start_percent
        self.total_playback_duration = len(audio_array) / playback_sample_rate # Total time of full canvas
        
        # 5. Play! (sounddevice plays non-blocking in the background automatically!)
        if DEBUG: print("Playing...")
        sd.play(audio_to_play, playback_sample_rate)
        
        # 6. Kick off the visual UI animation loop
        self.playback_start_time = time.time()
        self.update_playhead_loop()

    def stop_audio(self):
        """Immediately halts audio and resets the animation."""
        self.is_playing = False
        sd.stop()
        self.draw_playhead(self.time_slider.get() * self.width / 100)
        if DEBUG: print("Playback stopped.")

    def update_playhead_loop(self):
        """A recurring Tkinter loop that animates the playhead tracker."""
        if not self.is_playing:
            return
            
        # Calculate how many real-world seconds have passed since we hit play
        elapsed_time = time.time() - self.playback_start_time
        elapsed_percent = elapsed_time / self.total_playback_duration
        
        current_percent = self.playback_start_percent + elapsed_percent
        
        # Have we reached the end of the canvas?
        if current_percent >= 1.0:
            if self.loop_var.get():
                self.play_audio(from_loop=True) # Trigger standard loop!
            else:
                self.stop_audio() # Stop and leave playhead at the end
                self.draw_playhead(self.width)
            return

        # Move the playhead!
        current_x = int(current_percent * self.width)
        self.draw_playhead(current_x)
        
        # Tell Tkinter to run this function again in 30 milliseconds (approx 33 FPS)
        self.root.after(10, self.update_playhead_loop)

    def draw_playhead(self, x_pos):
        if self.playhead_triangle_id:
            self.canvas.delete(self.playhead_triangle_id)
        if self.playhead_line_id:
            self.canvas.delete(self.playhead_line_id)

        # Convert Matrix X to UI X
        ui_x_pos = (x_pos - self.pan_x) * self.zoom
        ui_x_pos += 1.45
                    
        # Hide it cleanly if it's panned completely off screen!
        if ui_x_pos < -20 or ui_x_pos > self.width + 20:
            return

        size = 8

        self.playhead_triangle_id = self.canvas.create_polygon(
            ui_x_pos - size, 0, ui_x_pos + size, 0, ui_x_pos, size + 5,
            fill="yellow", outline="black"
        )

        if self.show_playhead_line.get():
            self.playhead_line_id = self.canvas.create_line(
                ui_x_pos, size + 5, ui_x_pos, self.height,
                fill="yellow", dash=(4, 4)
            )

    def render_matrix_to_image(self):
        # 1. Handle Overlay Blending
        if hasattr(self, 'has_overlay') and self.has_overlay and self.import_mode.get() == "overlay":
            alpha = self.overlay_opacity.get()
            display_matrix = np.clip(self.amp_matrix + (self.overlay_matrix * alpha), 0.0, 1.0)
        else:
            display_matrix = self.amp_matrix
            
        # 2. Calculate Crop Bounds
        x1, y1 = int(self.pan_x), int(self.pan_y)
        x2 = int(self.pan_x + self.width / self.zoom)
        y2 = int(self.pan_y + self.height / self.zoom)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, max(x1+1, x2)), min(self.height, max(y1+1, y2))
        
        # 3. Map to colors
        cropped_matrix = display_matrix[y1:y2, x1:x2]
        colored_cropped = self.colormap(cropped_matrix) 
        cropped_int = (colored_cropped[:, :, :3] * 255).astype(np.uint8)
        
        # 4. Create the final images
        display_pil = Image.fromarray(cropped_int).resize((self.width, self.height), Image.Resampling.NEAREST)
        
        colored_full = self.colormap(display_matrix)
        self.pil_image = Image.fromarray((colored_full[:, :, :3] * 255).astype(np.uint8))
        
        self.tk_image = ImageTk.PhotoImage(image=display_pil)
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)

        if hasattr(self, 'show_grid'):
            self.update_grid()
            
        self.draw_axes()
        
    def start_stroke(self, event):
        self.undo_redo_counter += 1
        self.stop_audio()
        super().start_stroke(event)

    def undo(self, event=None):
        self.undo_redo_counter -= 1
        super().undo(event)
    
    def redo(self, event=None):
        self.undo_redo_counter += 1
        super().redo(event)

    def generate_ft_data(self):
        """Converts the visual UI drawing back into a High-Definition math array."""
        if hasattr(self, "has_overlay") and self.has_overlay and self.import_mode.get() == "overlay":
            ui_phase = getattr(self, "overlay_phase_matrix", None)
        else:
            ui_phase = getattr(self, "phase_matrix", None)
            
        audio_matrix = np.flipud(self.amp_matrix)
        
        duration_seconds = self.canvas_duration_seconds 
        sample_rate = self.canvas_sample_rate
        NFFT = (self.height - 1) * 2 
        ideal_step = NFFT // 2
        total_samples = int(sample_rate * duration_seconds)

        if self.zero_phase_matrix is None:
            self.zero_phase_matrix = np.zeros(audio_matrix.shape)
        if self.random_phase_matrix is None:
            self.random_phase_matrix = np.random.uniform(-np.pi, np.pi, size=audio_matrix.shape)
        if self.semi_random_phase_matrix is None:
            required_width = max(2, total_samples // ideal_step)
            
            times = np.linspace(0, duration_seconds, required_width)
            freqs = np.linspace(0, sample_rate / 2, self.height)
            
            # Mathematically synthesize perfectly continuous angles using (2 * pi * f * t) to completely avoid static
            self.semi_random_phase_matrix = 2 * np.pi * freqs[:, np.newaxis] * times[np.newaxis, :]
            
            # Add subtle random noise to soften pure shapes so they sound breathy
            randomness_factor = 0.5
            self.semi_random_phase_matrix += np.random.uniform(-np.pi, np.pi, size=self.semi_random_phase_matrix.shape) * randomness_factor
            self.semi_random_phase_matrix = (self.semi_random_phase_matrix + np.pi) % (2 * np.pi) - np.pi
        
        # --- PHASE ARCHITECTURE FIX ---
        pristine_phase = np.flipud(ui_phase if ui_phase is not None else self.semi_random_phase_matrix)
        
        if pristine_phase is not None:
            # 1. We imported an audio file! Force the Math matrix to precisely match the pristine phase!
            required_width = pristine_phase.shape[1]
            final_phase_matrix = pristine_phase
        else:
            # 2. We are drawing from scratch! Determine exactly how many chunks physics demands:
            final_phase_matrix = self.semi_random_phase_matrix
            
        # --- DELTA MAPPING ---
        # Are we synthesizing an imported "Edit" file?
        is_editing_imported_file = hasattr(self, 'pristine_amp_matrix') and self.import_mode.get() == "edit" and ui_phase is not None
        
        if is_editing_imported_file:
            # 1. Calculate ONLY the new edits the user made on the UI canvas!
            ui_delta = self.amp_matrix - getattr(self, 'base_squashed_amp')
            
            # 2. Stretch ONLY the edits up to the required physics width
            if ui_delta.shape[1] != required_width:
                pil_delta = Image.fromarray(ui_delta.astype(np.float32), mode="F")
                pil_delta = pil_delta.resize((required_width, self.height), Image.Resampling.BILINEAR)
                high_def_delta = np.array(pil_delta)
            else:
                high_def_delta = ui_delta
                
            # 3. Add the stretched edits onto the untouched, pristine background!
            audio_matrix = self.pristine_amp_matrix + high_def_delta
            audio_matrix = np.clip(audio_matrix, 0.0, 1.0)

            audio_matrix = np.flipud(audio_matrix)
        else:
            # We are drawing from scratch (or just synthesizing an Overlay drawing)
            audio_matrix = np.flipud(self.amp_matrix)
            
            if audio_matrix.shape[1] != required_width:
                pil_amp = Image.fromarray(audio_matrix.astype(np.float32), mode="F")
                pil_amp = pil_amp.resize((required_width, self.height), Image.Resampling.BILINEAR)
                audio_matrix = np.array(pil_amp)
            
        times = np.linspace(0, duration_seconds, required_width)
        freqs = np.linspace(0, sample_rate / 2, self.height)
        
        return FourierTransformData(
            amp_matrix=audio_matrix, phase_matrix=final_phase_matrix,
            freqs=freqs, times=times, sample_rate=sample_rate,
            NFFT=NFFT, noverlap=max(0, NFFT - ideal_step), useHanning=True
        )

    def save_file(self, event=None):
        """Exports the canvas to various formats, bypassing directory locks."""
        filepath = fd.asksaveasfilename(
            title="Export file",
            defaultextension=".wav",
            initialdir=os.getcwd(),
            filetypes=[("WAV Audio", "*.wav"), ("PNG Image", "*.png"), ("NumPy Array", "*.npz"), ("CSV Data", "*.csv")]
        )
        if not filepath: return
        
        ext = filepath.split('.')[-1].lower()
        
        if ext == "png":
            self.pil_image.save(filepath)
            mb.showinfo("Saved", f"Image saved to {filepath}")
            return
            
        # All other formats require mathematical generation
        ft_data = self.generate_ft_data()
        ft_data.apply_speed_change(self.speed_slider.get())
        
        if ext == "wav":
            ft_data.to_AudioSignal().to_audio_file(filepath, use_absolute_path=True)
            
        elif ext == "npz":
            filepath = filepath[:-4] + f" - Width-{self.width}, Height-{self.height}.npz"
            ft_data.to_npz_file(filepath, use_absolute_path=True)
            
        elif ext == "csv":
            ft_data.to_AudioSignal().to_csv_file(filepath, use_absolute_path=True)
            
        mb.showinfo("Saved", f"Data saved successfully to {filepath}")

    def open_file(self, event=None):
        """Imports data from any location, resizing it to perfectly fit the canvas."""
        filepath = fd.askopenfilename(
            title="Select data file",
            initialdir=os.getcwd(),
            filetypes=[("Audio/Fourier Transform data", "*.wav *.mp3 *.csv *.npz")]
        )
        if not filepath: return
        
        ext = filepath.split('.')[-1].lower()
        true_duration = None

        max_cache_time = datetime.timedelta(days=10)
        create_cache_data = True
        cached_file_dir = os.path.abspath(os.path.join("Data", "NPZs", "cached FT data"))
        cached_file_name = f"{os.path.basename(filepath)} - Width-{self.width}, Height-{self.height} - {datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.npz"
        cached_file_path = os.path.join(cached_file_dir, cached_file_name)
        
        for cached_file in os.listdir(cached_file_dir):
            if os.path.isfile(os.path.join(cached_file_dir, cached_file)) and cached_file[-4:] == ".npz":
                if datetime.datetime.now() - datetime.datetime.strptime(cached_file[:-4].split(" - ")[-1], "%Y-%m-%d %H-%M-%S") > max_cache_time:
                    os.remove(os.path.join(cached_file_dir, cached_file))
                elif " - ".join(cached_file_name.split(" - ")[:-2]) in cached_file and create_cache_data and\
                     filepath.split(".")[-1] != "npz" and\
                     cached_file.split(" - ")[1].split(", ")[-1] == f"Height-{self.height}":
                    cached_file_path = os.path.join(cached_file_dir, cached_file)
                    create_cache_data = False

        try:
            NFFT = (self.height - 1) * 2
            ideal_step = NFFT // 2

            if not create_cache_data:
                ft_data = FourierTransformData.from_npz_file(cached_file_path, use_absolute_path=True)
                # true_duration = AudioSignal.from_ft_data(ft_data).get_audio_duration()

            # 1. Extract the raw FT Data using raw library methods to bypass directory locks
            elif ext in ["wav", "mp3"]:
                signal = AudioSignal.from_audio_file(filepath, target_sample_rate=None, use_absolute_path=True)

                if len(signal.time_vs_amplitude) < NFFT + ideal_step:
                    signal.pad_zeros(NFFT + ideal_step - len(signal.time_vs_amplitude))
                
                true_duration = signal.get_audio_duration()
                ft_data = lib_rfft(signal, NFFT=NFFT, noverlap=max(0, NFFT - ideal_step), useHanning=True)
                
            elif ext == "csv":
                with open(filepath, "r") as f:
                    first_line = f.readline().strip()
                
                if first_line.startswith("sample_rate"):
                    if not filepath[:-4].split(" - ")[-1] == f"Width-{self.width}, Height-{self.height}":
                        mb.showerror("Import Error", f"Canvas height ({self.height}) does not match file height ({filepath[:-4].split(" - ")[-1].split(", ")[-1][len("Height-"):]})")
                        return
                    ft_data = FourierTransformData.from_csv_file(filepath, use_absolute_path=True)
                    create_cache_data = False
                    # true_duration = AudioSignal.from_ft_data(ft_data).get_audio_duration()
                
                elif first_line.startswith("Time"):
                    signal = AudioSignal.from_csv_file(filepath, use_absolute_path=True)

                    if len(signal.time_vs_amplitude) < NFFT + ideal_step:
                        signal.pad_zeros(NFFT + ideal_step - len(signal.time_vs_amplitude))
                    
                    true_duration = signal.get_audio_duration()
                    ft_data = lib_rfft(signal, NFFT=NFFT, noverlap=max(0, NFFT - ideal_step), useHanning=True)
                
                else:
                    raise ValueError("Unrecognized CSV format. Cannot determine if Audio or FFT data.")
                    
            elif ext == "npz":
                if not filepath[:-4].split(" - ")[-1] == f"Width-{self.width}, Height-{self.height}":
                    mb.showerror("Import Error", f"Canvas height ({self.height}) does not match file height ({filepath[:-4].split(" - ")[-1].split(", ")[-1][len("Height-"):]})")
                    return
                ft_data = FourierTransformData.from_npz_file(filepath, use_absolute_path=True)
                create_cache_data = False
                # true_duration = AudioSignal.from_ft_data(ft_data).get_audio_duration()

            # 2. Normalize and Prepare the Matrix
            imported_amp = ft_data.amp_matrix
            max_val = np.max(imported_amp)
            if max_val > 0:
                imported_amp = imported_amp / max_val
                
            # Flip Y axis (FT puts 0Hz at index 0, our UI puts 0Hz at bottom)
            imported_amp = np.flipud(imported_amp)

            # We MUST preserve and flip the phase too!
            imported_phase = np.flipud(ft_data.phase_matrix)

            # --- 3. Dynamic Canvas Stretching ---
            # We force BOTH axes to stretch to exactly fit your UI canvas!
            
            # 1. Amplitude (Visuals): Use BILINEAR so the colors blend smoothly
            pil_img = Image.fromarray((imported_amp * 255).astype(np.uint8))
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.BILINEAR)
            final_amp_matrix = np.array(pil_img).astype(float) / 255.0
            
            # 4. Apply based on user's mode preference
            if self.import_mode.get() == "edit":
                self.save_state()
                self.phase_matrix = imported_phase
                self.amp_matrix = final_amp_matrix.copy()
                self.pristine_amp_matrix = imported_amp.copy()
                self.base_squashed_amp = final_amp_matrix.copy()
            elif self.import_mode.get() == "overlay":
                self.overlay_matrix = final_amp_matrix
                self.overlay_phase_matrix = imported_phase
                self.has_overlay = True

            # --- 5. Sync the exact Physics ---
            self.canvas_sample_rate = int(ft_data.sample_rate)
            if true_duration:
                self.canvas_duration_seconds = float(true_duration)
            elif len(ft_data.times) > 1:
                self.canvas_duration_seconds = float(ft_data.times[-1] - ft_data.times[0])

            self.cached_base_audio = None

            self.time_slider.set(0)
            self.draw_playhead(0)
                
            self.render_matrix_to_image()
            
        except Exception as e:
            mb.showerror("Import Error", f"Failed to load file:\n{e}")

        if create_cache_data:
            ft_data = self.generate_ft_data()
            ft_data.to_npz_file(cached_file_path, use_absolute_path=True)


if __name__ == "__main__":
    root = tk.Tk()
    # app = DrawingProgram(root)
    app = SpectrogramSynthesizer(root, width=2000, height=400)
    root.mainloop()
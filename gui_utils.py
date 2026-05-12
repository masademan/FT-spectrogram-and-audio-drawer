import os
import time
import datetime
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
import sounddevice as sd
from PIL import Image, ImageTk
from scipy.ndimage import label
from utils import get_color_map
import tkinter.filedialog as fd
import tkinter.messagebox as mb
from fft_utils import AudioSignal, FourierTransformData, lib_rfft

DEBUG = True
AUDIO_PREGAIN_MULTIPLIER = 20.0
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
    def __init__(self, root, ui_width=1000, ui_height=400, res_width=2000, res_height=513, name="Drawing Program"):
        self.root = root
        self.root.title(name)
        
        self.ui_width = ui_width
        self.ui_height = ui_height
        self.width = res_width
        self.height = res_height

        self.max_diagonal = int((self.width**2 + self.height**2)**0.5) + 1
        self.amp_matrix = np.zeros((self.height, self.width), dtype=float)

        # --- The Native Phase Layer! (0.5 represents 0 Radians) ---
        self.phase_matrix = np.full((self.height, self.width), 0.5, dtype=float)
        self.view_mode = tk.StringVar(value="Amplitude")
        self.phase_is_edited = False
        
        # --- NEW L-SHAPED MASTER LAYOUT ---
        self.master_frame = tk.Frame(root)
        self.master_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # TOP HALF: Canvas (Left) + Drawing Tools (Right)
        self.top_pane = tk.Frame(self.master_frame)
        self.top_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 1. CANVAS WORKSPACE (Left)
        self.canvas_pane = tk.Frame(self.top_pane)
        self.canvas_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.workspace_frame = tk.Frame(self.canvas_pane)
        self.workspace_frame.pack(pady=10, anchor=tk.N)
        
        # Y-Axis (Uses UI height)
        self.y_axis_canvas = tk.Canvas(self.workspace_frame, width=80, height=self.ui_height, bg=root.cget("bg"), highlightthickness=0)
        self.y_axis_canvas.grid(row=0, column=0, sticky="ns")
        
        # Main Drawing Board (Uses UI dimensions)
        self.canvas = tk.Canvas(self.workspace_frame, width=self.ui_width, height=self.ui_height, bg="black", cursor="crosshair")
        self.canvas.grid(row=0, column=1)
        
        # X-Axis (Uses UI width)
        self.x_axis_canvas = tk.Canvas(self.workspace_frame, width=self.ui_width, height=30, bg=root.cget("bg"), highlightthickness=0)
        self.x_axis_canvas.grid(row=1, column=1, sticky="ew")

        # 2. DRAWING TOOLS SIDEBAR (Right)
        self.drawing_tools_pane = tk.Frame(self.top_pane)
        self.drawing_tools_pane.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # --- PAN & ZOOM STATE VARIABLES ---
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_pan_x = None
        self.last_pan_y = None
        
        self.tk_image = None 
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        
        # 1. BRUSH & OPACITY FRAME
        self.brush_frame = tk.LabelFrame(self.drawing_tools_pane, text="Brush & Opacity")
        self.brush_frame.pack(side=tk.TOP, pady=5, fill=tk.X)
        
        self.opacity_slider = tk.Scale(self.brush_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=120)
        self.opacity_slider.set(0.5) 
        self.opacity_slider.pack(side=tk.TOP, padx=5)
        
        self.negate_opacity = tk.BooleanVar(value=False)
        tk.Checkbutton(self.brush_frame, text="Erase", variable=self.negate_opacity).pack(side=tk.TOP)

        def enforce_brush_resolution(val):
            v = float(val)
            if v > 5.0 and v != round(v):
                self.brush_slider.set(round(v))
                
        self.brush_slider = tk.Scale(self.brush_frame, from_=0.5, to=50, resolution=0.5, orient=tk.HORIZONTAL, length=120, command=enforce_brush_resolution)
        self.brush_slider.set(4) 
        self.brush_slider.pack(side=tk.TOP, padx=5)

        self.show_brush_outline = tk.BooleanVar(value=True)
        tk.Checkbutton(self.brush_frame, text="Show Brush", variable=self.show_brush_outline, command=self.toggle_brush_outline).pack(side=tk.TOP)
        
        self.brush_outline_id = None 
        self.show_brush_while_drawing = tk.BooleanVar(value=False)
        tk.Checkbutton(self.brush_frame, text="Show Brush While Drawing", variable=self.show_brush_while_drawing).pack(side=tk.TOP)

        # 2. VIEW & GRID FRAME
        self.view_frame = tk.LabelFrame(self.drawing_tools_pane, text="View & Grid")
        self.view_frame.pack(side=tk.TOP, pady=5, fill=tk.X)
        
        view_top_frame = tk.Frame(self.view_frame)
        view_top_frame.pack(side=tk.TOP, fill=tk.X)

        self.brightness_slider = tk.Scale(view_top_frame, from_=-1.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=100, label="Brightness", command=lambda _: self.render_matrix_to_image())
        self.brightness_slider.set(0.0)
        self.brightness_slider.pack(side=tk.LEFT, padx=5)
        
        btn_col = tk.Frame(view_top_frame)
        btn_col.pack(side=tk.LEFT, padx=5)
        tk.Button(btn_col, text="Reset Bright", command=lambda: self.brightness_slider.set(0.0)).pack(side=tk.TOP, pady=2)
        tk.Button(btn_col, text="🏠 Home", command=self.reset_view).pack(side=tk.TOP, pady=2)

        self.grid_x_slider = tk.Scale(self.view_frame, from_=1, to=50, orient=tk.HORIZONTAL, length=100, label="Time (X) Density", command=lambda _: self.render_matrix_to_image())
        self.grid_x_slider.set(10)
        self.grid_x_slider.pack(side=tk.TOP)

        self.grid_y_slider = tk.Scale(self.view_frame, from_=1, to=50, orient=tk.HORIZONTAL, length=100, label="Freq (Y) Density", command=lambda _: self.render_matrix_to_image())
        self.grid_y_slider.set(10)
        self.grid_y_slider.pack(side=tk.TOP)

        self.show_grid = tk.BooleanVar(value=False)
        tk.Checkbutton(self.view_frame, text="Show Grid", variable=self.show_grid, command=self.render_matrix_to_image).pack(side=tk.TOP)

        # 3. TOOLS FRAME (2x4 Grid setup!)
        self.tools_frame = tk.LabelFrame(self.drawing_tools_pane, text="Tools")
        self.tools_frame.pack(side=tk.TOP, pady=5, fill=tk.X)
        
        self.current_tool = tk.StringVar(value="pencil")
        tools = [
            ("Pencil", "pencil"), ("Non-Add", "non_add"), ("TBD", "new_tool"), ("Dropper", "dropper"),
            ("Rectangle", "square"), ("Circle", "circle"), ("Line", "line"), ("Bucket", "bucket")
        ]
        
        row1 = tk.Frame(self.tools_frame)
        row1.pack(side=tk.TOP, pady=5)
        for text, val in tools[:4]:
            tk.Radiobutton(row1, text=text, variable=self.current_tool, value=val, indicatoron=0, width=8).pack(side=tk.LEFT, padx=2)
            
        row2 = tk.Frame(self.tools_frame)
        row2.pack(side=tk.TOP, pady=5)
        for text, val in tools[4:]:
            tk.Radiobutton(row2, text=text, variable=self.current_tool, value=val, indicatoron=0, width=8).pack(side=tk.LEFT, padx=2)

        # 4. HISTORY & CANVAS FRAME
        self.history_frame = tk.LabelFrame(self.drawing_tools_pane, text="History & Colors")
        self.history_frame.pack(side=tk.TOP, pady=5, fill=tk.X)
        
        self.cmap_var = tk.StringVar(value="spectral_v3")
        self.cmap_dropdown = ttk.Combobox(self.history_frame, textvariable=self.cmap_var, 
                                          values=list(COLOR_DESCRIPTIONS.keys()), state="readonly", width=15)
        self.cmap_dropdown.pack(side=tk.TOP, pady=2)
        
        self.cmap_desc_label = tk.Label(self.history_frame, text=COLOR_DESCRIPTIONS["spectral_v3"], 
                                        font=("Arial", 8, "italic"), fg="gray", wraplength=130, justify="center")
        self.cmap_desc_label.pack(side=tk.TOP)

        self.max_history = 25 
        self.undo_stack = []
        self.redo_stack = []
        
        self.history_btn_frame = tk.Frame(self.history_frame)
        self.history_btn_frame.pack(side=tk.TOP, pady=2)
        
        tk.Button(self.history_btn_frame, text="⬅ Undo", command=self.undo, width=6).pack(side=tk.LEFT, padx=2)
        tk.Button(self.history_btn_frame, text="Redo ➡", command=self.redo, width=6).pack(side=tk.LEFT, padx=2)

        tk.Button(self.history_frame, text="🗑 Clear Canvas", bg="#ffcccc", command=self.clear_canvas).pack(side=tk.TOP, pady=5, fill=tk.X, padx=10)

        # --- SHAPE TRACKING VARIABLES ---
        self.shape_anchor_x = None
        self.shape_anchor_y = None
        self.temp_shape_id = None

        # --- KEYBOARD SHORTCUTS ---
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Control-Z>", self.redo) 
        self.root.bind("<Control-w>", self.close_window)
        self.root.bind("<Escape>", self.close_window)

        # --- PAN & ZOOM BINDINGS ---
        self.canvas.bind("<MouseWheel>", self.zoom_canvas)
        self.canvas.bind("<Button-4>", self.zoom_canvas) 
        self.canvas.bind("<Button-5>", self.zoom_canvas) 
        self.canvas.bind("<ButtonPress-3>", self.start_pan) 
        self.canvas.bind("<B3-Motion>", self.do_pan)
            
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

    def close_window(self, event=None):
        self.root.destroy()

    def zoom_canvas(self, event):
        ui_x_norm = event.x / self.ui_width
        ui_y_norm = event.y / self.ui_height
        
        mx_before = self.pan_x + (ui_x_norm * (self.width / self.zoom))
        my_before = self.pan_y + (ui_y_norm * (self.height / self.zoom))
        
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.zoom = min(self.zoom * 1.2, 50.0) 
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.zoom = max(self.zoom / 1.2, 1.0)  
            
        self.pan_x = mx_before - (ui_x_norm * (self.width / self.zoom))
        self.pan_y = my_before - (ui_y_norm * (self.height / self.zoom))
        
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
        
        dx_ui_norm = (event.x - self.last_pan_x) / self.ui_width
        dy_ui_norm = (event.y - self.last_pan_y) / self.ui_height
        
        self.pan_x -= dx_ui_norm * (self.width / self.zoom)
        self.pan_y -= dy_ui_norm * (self.height / self.zoom)

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
        
        sr = getattr(self, 'canvas_sample_rate', 44100)
        duration = getattr(self, 'canvas_duration_seconds', 5.0)
        
        x1, y1, x2, y2 = self.get_crop_bounds()
        
        density_y = self.grid_y_slider.get()
        for i in range(density_y + 1):
            ui_y = i * (self.ui_height / density_y)
            ui_y_norm = ui_y / self.ui_height
            my = y1 + (ui_y_norm * (y2 - y1))
            freq = ((self.height - my) / self.height) * (sr / 2) 
            
            if 0 <= ui_y <= self.ui_height:
                if i == 0: anchor_pos = "ne"      
                elif i == density_y: anchor_pos = "se"   
                else: anchor_pos = "e"            
                self.y_axis_canvas.create_text(75, ui_y, text=f"{int(freq)} Hz", anchor=anchor_pos, fill="gray", font=("Arial", 8))
                self.y_axis_canvas.create_line(75, ui_y, 80, ui_y, fill="gray")

        density_x = self.grid_x_slider.get()
        for i in range(density_x + 1):
            ui_x = i * (self.ui_width / density_x)
            ui_x_norm = ui_x / self.ui_width
            mx = x1 + (ui_x_norm * (x2 - x1))
            time_val = (mx / self.width) * duration
            
            if 0 <= ui_x <= self.ui_width:
                if i == 0: anchor_pos = "nw"      
                elif i == density_x: anchor_pos = "ne"   
                else: anchor_pos = "n"            
                self.x_axis_canvas.create_text(ui_x, 15, text=f"{time_val:.2f}s", anchor=anchor_pos, fill="gray", font=("Arial", 8))
                self.x_axis_canvas.create_line(ui_x, 0, ui_x, 5, fill="gray")

    def save_state(self):
        """Saves a snapshot of the current matrix to the undo stack."""
        self.undo_stack.append((self.amp_matrix.copy(), self.phase_matrix.copy()))
        
        # Enforce the stack size limit so we don't blow out the RAM!
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0) 
            
        # The moment you draw something new, the redo "future" is destroyed
        self.redo_stack.clear() 

    def undo(self, event=None):
        if self.undo_stack:
            # Save the current state to the redo stack before going back
            self.redo_stack.append((self.amp_matrix.copy(), self.phase_matrix.copy()))
            
            # Pop the last saved state and make it the active matrix
            self.amp_matrix, self.phase_matrix = self.undo_stack.pop()
            self.render_matrix_to_image()

    def redo(self, event=None):
        if self.redo_stack:
            # Save the current state to the undo stack before going forward
            self.undo_stack.append((self.amp_matrix.copy(), self.phase_matrix.copy()))
            
            # Pop the future state and make it the active matrix
            self.amp_matrix, self.phase_matrix = self.redo_stack.pop()
            self.render_matrix_to_image()

    def clear_canvas(self):
        """Saves history and completely clears the main drawing canvas."""
        self.save_state()

        if self.view_mode.get() == "Amplitude":
            self.amp_matrix.fill(0.0)
        else:
            self.phase_matrix.fill(0.5)
            self.phase_is_edited = True

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

        # Translate High-Def Matrix coords back to UI pixels
        x1, y1, x2, y2 = self.get_crop_bounds()
        ui_x = ((x - x1) / (x2 - x1)) * self.ui_width
        ui_y = ((y - y1) / (y2 - y1)) * self.ui_height
        
        ui_r = r # * self.zoom

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
        active_mat = self.amp_matrix if self.view_mode.get() == "Amplitude" else self.phase_matrix
        
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
                
                distance = np.hypot(x - self.last_x, y - self.last_y)
                steps = max(1, int(distance * 4)) # 4x steps for ultra-smooth sub-pixel lines
                
                matrix_rx = max(0.5, (brush_radius / self.zoom) * (self.width / self.ui_width))
                matrix_ry = max(0.5, (brush_radius / self.zoom) * (self.height / self.ui_height))

                for i in range(1, steps + 1):
                    t = i / steps
                    interp_x = self.last_x + (x - self.last_x) * t
                    interp_y = self.last_y + (y - self.last_y) * t
                    
                    if 0 <= interp_x < self.width and 0 <= interp_y < self.height:
                        y_start = max(0, int(interp_y - matrix_ry))
                        y_end = min(self.height, int(interp_y + matrix_ry) + 1)
                        x_start = max(0, int(interp_x - matrix_rx))
                        x_end = min(self.width, int(interp_x + matrix_rx) + 1)
                        
                        for py in range(y_start, y_end):
                            for px in range(x_start, x_end):
                                if ((px - interp_x) / matrix_rx)**2 + ((py - interp_y) / matrix_ry)**2 <= 1.0:
                                    if (px, py) not in temp_drawn_pixels:
                                        active_mat[py, px] += intensity
                                        temp_drawn_pixels.add((px, py))
            
            elif tool == "square":
                intensity = self.opacity_slider.get() * (-1 if self.negate_opacity.get() else 1)
                
                matrix_rx = max(0.5, (brush_radius / self.zoom) * (self.width / self.ui_width))
                matrix_ry = max(0.5, (brush_radius / self.zoom) * (self.height / self.ui_height))

                Tx = int(max(1, matrix_rx * 2))
                Ty = int(max(1, matrix_ry * 2))
                
                x1, x2 = sorted([self.shape_anchor_x, x])
                y1, y2 = sorted([self.shape_anchor_y, y])
                
                x1 = int(max(0, x1 - Tx/2))
                x2 = int(min(self.width, x2 + Tx/2))
                y1 = int(max(0, y1 - Ty/2))
                y2 = int(min(self.height, y2 + Ty/2))

                y_inner_top = int(min(y1 + Ty, y2))
                y_inner_bottom = int(max(y1 + Ty, y2 - Ty))
                x_inner_left = int(min(x1 + Tx, x2))
                x_inner_right = int(max(x1 + Tx, x2 - Tx))
                
                active_mat[y1:y_inner_top, x1:x2] += intensity
                active_mat[y_inner_bottom:y2, x1:x2] += intensity
                active_mat[y_inner_top:y_inner_bottom, x1:x_inner_left] += intensity
                active_mat[y_inner_top:y_inner_bottom, x_inner_right:x2] += intensity

            elif tool == "bucket":
                intensity = self.opacity_slider.get() if not self.negate_opacity.get() else 0
                safe_x = int(max(0, min(self.width - 1, x)))
                safe_y = int(max(0, min(self.height - 1, y)))
                target_amp = active_mat[safe_y, safe_x]
                matching_pixels_mask = np.abs(active_mat - target_amp) < 1e-4
                labeled_matrix, _ = label(matching_pixels_mask)
                clicked_blob_id = labeled_matrix[safe_y, safe_x]
                active_mat[labeled_matrix == clicked_blob_id] = intensity

            elif tool == "circle":
                intensity = self.opacity_slider.get()
                
                matrix_rx = max(0.5, (brush_radius / self.zoom) * (self.width / self.ui_width))
                matrix_ry = max(0.5, (brush_radius / self.zoom) * (self.height / self.ui_height))
                
                x1, x2 = sorted([self.shape_anchor_x, x])
                y1, y2 = sorted([self.shape_anchor_y, y])
                
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                rx = max(0.1, (x2 - x1) / 2.0)
                ry = max(0.1, (y2 - y1) / 2.0)
                
                box_x1 = max(0, int(cx - rx - matrix_rx - 1))
                box_x2 = min(self.width, int(cx + rx + matrix_rx + 1))
                box_y1 = max(0, int(cy - ry - matrix_ry - 1))
                box_y2 = min(self.height, int(cy + ry + matrix_ry + 1))
                
                if box_x2 > box_x1 and box_y2 > box_y1:
                    Y, X = np.ogrid[box_y1:box_y2, box_x1:box_x2]
                    
                    outer_mask = ((X - cx) / (rx + matrix_rx))**2 + ((Y - cy) / (ry + matrix_ry))**2 <= 1.0
                    
                    inner_rx = max(0.0001, rx - matrix_rx)
                    inner_ry = max(0.0001, ry - matrix_ry)
                    inner_mask = ((X - cx) / inner_rx)**2 + ((Y - cy) / inner_ry)**2 <= 1.0
                    
                    ring_mask = outer_mask & ~inner_mask
                    active_mat[box_y1:box_y2, box_x1:box_x2][ring_mask] += intensity

            # Re-render the matrix to show the baked shape
            if self.view_mode.get() == "Amplitude":
                self.amp_matrix = np.clip(self.amp_matrix, 0.0, 1.0)
            else:
                self.phase_matrix = np.clip(self.phase_matrix, 0.0, 1.0)

            self.render_matrix_to_image()
        

        # Clean up variables for the next stroke
        self.last_x, self.last_y = None, None
        self.currently_clicking = False
        self.shape_anchor_x, self.shape_anchor_y = None, None
        self.drawn_pixels.clear() 
        self.pixel_history.clear()

        self.hide_brush_outline()
        self.hover(event)

    def get_crop_bounds(self):
        """Calculates the exact integer bounds of the visible matrix slice."""
        x1, y1 = int(self.pan_x), int(self.pan_y)
        x2 = int(self.pan_x + self.width / self.zoom)
        y2 = int(self.pan_y + self.height / self.zoom)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, max(x1+1, x2)), min(self.height, max(y1+1, y2))
        return x1, y1, x2, y2

    def get_modified_coords(self, event, tool):
        ui_x_norm = event.x / self.ui_width
        ui_y_norm = event.y / self.ui_height
        
        # --- FEATURE: Snap to Visual Grid Before Converting to Matrix Coords! ---
        if event.state & 0x0004:
            density_x = self.grid_x_slider.get()
            density_y = self.grid_y_slider.get()
            
            ui_x_step_norm = 1.0 / density_x
            ui_y_step_norm = 1.0 / density_y
            
            ui_x_norm = round(ui_x_norm / ui_x_step_norm) * ui_x_step_norm
            ui_y_norm = round(ui_y_norm / ui_y_step_norm) * ui_y_step_norm

        x1, y1, x2, y2 = self.get_crop_bounds()
        matrix_x = x1 + (ui_x_norm * (x2 - x1))
        matrix_y = y1 + (ui_y_norm * (y2 - y1))
        
        x, y = matrix_x, matrix_y

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

        return x, y

    def update_grid(self):
        self.canvas.delete("grid_line")
        if self.show_grid.get():
            density_x = self.grid_x_slider.get()
            density_y = self.grid_y_slider.get()
            
            # Start at 1 so we don't draw lines exactly on the outer borders
            for i in range(1, density_x): 
                ui_x = i * (self.ui_width / density_x)
                self.canvas.create_line(ui_x, 0, ui_x, self.ui_height, fill="gray", dash=(2, 4), tags="grid_line")
                
            for i in range(1, density_y):
                ui_y = i * (self.ui_height / density_y)
                self.canvas.create_line(0, ui_y, self.ui_width, ui_y, fill="gray", dash=(2, 4), tags="grid_line")

    def paint(self, event):
        x, y = event.x, event.y
        tool = self.current_tool.get()
        opacity = self.opacity_slider.get()

        if self.show_brush_while_drawing.get():
            self.hover(event)
        
        # Get the modified coordinates!
        x, y = self.get_modified_coords(event, tool)

        x1, y1, x2, y2 = self.get_crop_bounds()

        brush_radius = self.brush_slider.get()

        # --- SHAPE RENDERING (Visual Only) ---
        if tool in ["square", "circle", "line", "bucket", "dropper"]:
            ui_anchor_x = ((self.shape_anchor_x - x1) / (x2 - x1)) * self.ui_width
            ui_anchor_y = ((self.shape_anchor_y - y1) / (y2 - y1)) * self.ui_height
            ui_x = ((x - x1) / (x2 - x1)) * self.ui_width
            ui_y = ((y - y1) / (y2 - y1)) * self.ui_height
            
            if tool == "line":
                if self.temp_shape_id:
                    self.canvas.coords(self.temp_shape_id, ui_anchor_x, ui_anchor_y, ui_x, ui_y)
                else:
                    # If it doesn't exist yet, create it!
                    self.temp_shape_id = self.canvas.create_line(
                        ui_anchor_x, ui_anchor_y, ui_x, ui_y, 
                        fill="red", width=4, dash=(4, 4)
                    )
                    
            elif tool == "square":
                if self.temp_shape_id:
                    self.canvas.coords(self.temp_shape_id, ui_anchor_x, ui_anchor_y, ui_x, ui_y)
                else:
                    self.temp_shape_id = self.canvas.create_rectangle(
                        ui_anchor_x, ui_anchor_y, ui_x, ui_y, 
                        outline="red", width=4, dash=(4, 4)
                    )
                    
            elif tool == "circle":
                if self.temp_shape_id:
                    self.canvas.coords(self.temp_shape_id, ui_anchor_x, ui_anchor_y, ui_x, ui_y)
                else:
                    self.temp_shape_id = self.canvas.create_oval(
                        ui_anchor_x, ui_anchor_y, ui_x, ui_y, 
                        outline="red", width=4, dash=(4, 4)
                    )
                    
            elif tool == "dropper":
                safe_x = max(0, min(self.width - 1, int(x)))
                safe_y = max(0, min(self.height - 1, int(y)))
                active_mat = self.amp_matrix if self.view_mode.get() == "Amplitude" else self.phase_matrix
                sampled_amplitude = active_mat[safe_y, safe_x]
                self.opacity_slider.set(sampled_amplitude)
                
            return # Exit the function early!
        
        # Get the active matrix!
        active_mat = self.amp_matrix if self.view_mode.get() == "Amplitude" else self.phase_matrix
        if self.view_mode.get() == "Phase":
            self.phase_is_edited = True

        if self.last_x is None:
            self.last_x, self.last_y = x, y
        
        intensity = -opacity if self.negate_opacity.get() else opacity
        
        distance = np.hypot(x - self.last_x, y - self.last_y)
        steps = max(1, int(distance * 4))
        
        for i in range(1, steps + 1):
            t = i / steps
            interp_x = self.last_x + (x - self.last_x) * t
            interp_y = self.last_y + (y - self.last_y) * t
            
            self.stroke_distance += (distance / steps)

            matrix_rx = max(0.5, (brush_radius / self.zoom) * (self.width / self.ui_width))
            matrix_ry = max(0.5, (brush_radius / self.zoom) * (self.height / self.ui_height))
            
            if 0 <= interp_x < self.width and 0 <= interp_y < self.height:
                y_start = max(0, int(interp_y - matrix_ry))
                y_end = min(self.height, int(interp_y + matrix_ry) + 1)
                x_start = max(0, int(interp_x - matrix_rx))
                x_end = min(self.width, int(interp_x + matrix_rx) + 1)
                
                for py in range(y_start, y_end):
                    for px in range(x_start, x_end):
                        if ((px - interp_x) / matrix_rx)**2 + ((py - interp_y) / matrix_ry)**2 <= 1.0:
                            
                            if tool == "non_add":
                                if (px, py) not in self.drawn_pixels:
                                    active_mat[py, px] += intensity
                                    self.drawn_pixels.add((px, py))
                            else: 
                                last_painted_dist = self.pixel_history.get((px, py), -self.max_diagonal)
                                # Tie the gap spacing to the dynamic matrix radius!
                                if self.stroke_distance - last_painted_dist > (matrix_rx * 2):
                                    active_mat[py, px] += intensity
                                    self.pixel_history[(px, py)] = self.stroke_distance

        if self.view_mode.get() == "Amplitude":
            self.amp_matrix = np.clip(self.amp_matrix, 0.0, 1.0)
        else:
            self.phase_matrix = np.clip(self.phase_matrix, 0.0, 1.0)

        self.last_x, self.last_y = x, y
        self.render_matrix_to_image()

    def render_matrix_to_image(self):
        """The Render Engine: Crops the matrix based on zoom, applies brightness, and upscales it to the canvas."""
        x1, y1, x2, y2 = self.get_crop_bounds()
        
        # Check active view mode!
        viewing_phase = getattr(self, 'view_mode', tk.StringVar(value="Amplitude")).get() == "Phase"
        # Route the data arrays dynamically
        active_main = self.phase_matrix if viewing_phase else self.amp_matrix
        cropped_main = active_main[y1:y2, x1:x2]
        
        # --- Add Brightness Offset Visually ---
        brightness = self.brightness_slider.get()

        if brightness != 0.0:
            display_slice = np.clip(cropped_main + brightness, 0.0, 1.0)
            full_display = np.clip(self.amp_matrix + brightness, 0.0, 1.0)
        else:
            display_slice = cropped_main
            full_display = self.amp_matrix
            
        colored_cropped = self.colormap(display_slice)
        cropped_int = (colored_cropped[:, :, :3] * 255).astype(np.uint8)
        display_pil = Image.fromarray(cropped_int).resize((self.ui_width, self.ui_height), Image.Resampling.NEAREST)
        
        colored_full = self.colormap(full_display)
        self.pil_image = Image.fromarray((colored_full[:, :, :3] * 255).astype(np.uint8))
        
        self.tk_image = ImageTk.PhotoImage(image=display_pil)
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)

        self.update_grid()
        self.draw_axes()

class SpectrogramSynthesizer(AudioDrawingProgram):
    def __init__(self, root, ui_width=1000, ui_height=400, res_width=2000, res_height=513, name="Spectrogram DAW"):
        # Initialize the base drawing program!
        super().__init__(root, ui_width, ui_height, res_width, res_height, name)

        self.canvas_sample_rate = 44100
        self.canvas_duration_seconds = 5.0
        self.original_max_val = float(self.height - 1)

        # --- BOTTOM PANE FOR SYNTHESIZER TOOLS ---
        self.bottom_pane = tk.Frame(self.master_frame)
        self.bottom_pane.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # 1. FILE I/O & OVERLAY FRAME
        self.io_frame = tk.LabelFrame(self.bottom_pane, text="File I/O & Overlay")
        self.io_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        tk.Button(self.io_frame, text="📂 Open (Ctrl+O)", command=self.open_file).pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        tk.Button(self.io_frame, text="💾 Save (Ctrl+S)", command=self.save_file).pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        self.root.bind("<Control-o>", self.open_file)
        self.root.bind("<Control-s>", self.save_file)
        
        self.import_mode = tk.StringVar(value="edit")
        tk.Radiobutton(self.io_frame, text="Import to Edit", variable=self.import_mode, value="edit", command=self.render_matrix_to_image).pack(side=tk.TOP, anchor=tk.W)
        tk.Radiobutton(self.io_frame, text="Import as Overlay", variable=self.import_mode, value="overlay", command=self.render_matrix_to_image).pack(side=tk.TOP, anchor=tk.W)
        
        # --- View & Trace Toggles ---
        self.show_overlay = tk.BooleanVar(value=True)
        tk.Checkbutton(self.io_frame, text="Show Overlay", variable=self.show_overlay, command=self.render_matrix_to_image).pack(side=tk.TOP, anchor=tk.W)

        self.draw_in_red = tk.BooleanVar(value=True)
        tk.Checkbutton(self.io_frame, text="Red Draw Mode", variable=self.draw_in_red, command=self.render_matrix_to_image).pack(side=tk.TOP, anchor=tk.W)
        
        tk.Button(self.io_frame, text="🗑 Clear Overlay", bg="#ffcccc", command=self.clear_overlay).pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # --- Swap Canvas & Overlay Button ---
        tk.Button(self.io_frame, text="🔄 Swap Canvas/Overlay", command=self.swap_canvas_overlay).pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        self.overlay_opacity = tk.Scale(self.io_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, label="Overlay Opacity", command=lambda _: self.render_matrix_to_image())
        self.overlay_opacity.set(0.4)
        self.overlay_opacity.pack(side=tk.TOP, padx=5)

        view_top_frame = tk.Frame(self.view_frame)
        view_top_frame.pack(side=tk.TOP, fill=tk.X)
        
        # --- Dynamic View Toggles ---
        view_toggle_frame = tk.Frame(view_top_frame)
        view_toggle_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        tk.Radiobutton(view_toggle_frame, text="View Amp", variable=self.view_mode, value="Amplitude", command=self.render_matrix_to_image).pack(side=tk.LEFT)
        tk.Radiobutton(view_toggle_frame, text="View Phase", variable=self.view_mode, value="Phase", command=self.render_matrix_to_image).pack(side=tk.LEFT)
        
        # Overlay State
        self.overlay_matrix = np.zeros((self.height, self.width), dtype=float)
        self.has_overlay = False

        self.zero_phase_matrix = None
        self.random_phase_matrix = None
        self.semi_random_phase_matrix = None

        # 2. PLAYBACK ENGINE FRAME
        self.playback_frame = tk.LabelFrame(self.bottom_pane, text="Playback Engine")
        self.playback_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        btn_frame = tk.Frame(self.playback_frame)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        self.play_btn = tk.Button(btn_frame, text="▶ Play", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=8, command=self.toggle_play_pause)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop", bg="#f44336", fg="white", font=("Arial", 10, "bold"), width=8, command=self.stop_audio)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        # --- Audio Source Routing Dropdown! ---
        self.prev_playback_source = "Both (Mix)"
        tk.Label(self.playback_frame, text="Audio Source:").pack(side=tk.TOP, pady=(5, 0))
        self.playback_source = tk.StringVar(value=self.prev_playback_source)
        self.playback_source.trace_add("write", self.clear_audio_cache)
        
        ttk.Combobox(self.playback_frame, textvariable=self.playback_source, values=["Main Canvas", "Overlay / Original", "Both (Mix)"], state="readonly", width=16).pack(side=tk.TOP, padx=5, pady=2)

        # --- STREAM ENGINE STATE VARIABLES ---
        self.playback_stream = None
        self.playback_index = 0
        self.total_playback_samples = 0
        self.cached_samples_ready = 0
        self.synthesis_flag = 0 # Unique ID to manage background threads!
        # -------------------------------------
        
        self.is_playing = False
        self.playback_start_time = 0.0
        self.playback_start_percent = 0.0
        self.total_playback_duration = 0.0

        self.undo_redo_counter = 0
        
        self.loop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_frame, text="Loop", variable=self.loop_var).pack(side=tk.LEFT, padx=5)
        
        self.time_slider = tk.Scale(self.playback_frame, from_=0.0, to=100.0, resolution=0.1, orient=tk.HORIZONTAL, label="Start Position (%)", length=300, command=lambda _: self.draw_playhead(self.time_slider.get() * self.width / 100, True))
        self.time_slider.pack(side=tk.TOP, padx=5)
        
        self.show_playhead_line = tk.BooleanVar(value=True)
        tk.Checkbutton(btn_frame, text="Show Playhead Line", variable=self.show_playhead_line, command=lambda: self.draw_playhead(self.time_slider.get() * self.width / 100)).pack(side=tk.LEFT, padx=5)

        # 3. SETTINGS FRAME
        self.settings_frame = tk.LabelFrame(self.bottom_pane, text="Settings")
        self.settings_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        # --- Thread-safe Volume Tracker ---
        self.current_volume = 1.0 
        
        vol_frame = tk.Frame(self.settings_frame)
        vol_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Add the 'command' hook!
        self.volume_slider = tk.Scale(vol_frame, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, label="Volume", command=self.update_volume_tracker)
        self.volume_slider.set(1.0)
        self.volume_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(vol_frame, text="↺", font=("Arial", 10, "bold"), command=lambda: self.volume_slider.set(1.0)).pack(side=tk.RIGHT, padx=2, pady=(15, 0))
        
        # --- Speed Slider with Reset Button ---
        speed_frame = tk.Frame(self.settings_frame)
        speed_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.speed_slider = tk.Scale(speed_frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label="Speed")
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(speed_frame, text="↺", font=("Arial", 13, "bold"), command=lambda: self.speed_slider.set(1.0)).pack(side=tk.RIGHT, padx=2, pady=(15, 0))

        # self.phase_gen_mode = tk.StringVar(value="Original/Semi-Random")
        self.phase_gen_mode = tk.StringVar(value="Semi-Random Phase")
        tk.Label(self.settings_frame, text="Synthesis Phase").pack(side=tk.TOP, padx=5, pady=(5,0))
        # ttk.Combobox(self.settings_frame, textvariable=self.phase_gen_mode, values=["Original/Semi-Random", "Pure Random", "Zero (Robotic)"], state="readonly", width=18).pack(side=tk.TOP, padx=5, pady=2)
        ttk.Combobox(self.settings_frame, textvariable=self.phase_gen_mode, values=["0 Phase", "Random Phase", "Semi-Random Phase", "Saved Phase"], state="readonly", width=18).pack(side=tk.TOP, padx=5, pady=2)
        tk.Button(self.settings_frame, text="Apply Phase", command=self.apply_phase_generator).pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # 4. PHASE TOOLS FRAME
        # self.phase_frame = tk.LabelFrame(self.bottom_pane, text="Phase Tools")
        # self.phase_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        # self.phase_gen_mode = tk.StringVar(value="Semi-Random Phase")
        # ttk.Combobox(self.phase_frame, textvariable=self.phase_gen_mode, values=["0 Phase", "Random Phase", "Semi-Random Phase", "Saved Phase"], state="readonly", width=16).pack(side=tk.TOP, padx=5, pady=2)
        # tk.Button(self.phase_frame, text="Apply Phase", command=self.apply_phase_generator).pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # --- KEYBOARD SHORTCUTS ---
        self.root.bind("<space>", self.toggle_play_pause)
        
        # PLAYHEAD VISUALS
        self.playhead_triangle_id = None
        self.playhead_line_id = None
        self.draw_playhead(0) # Initialize at 0

    def update_volume_tracker(self, val):
        """Safely caches the UI slider so the background audio thread can read it without crashing Tkinter!"""
        self.current_volume = float(val)

    def toggle_play_pause(self, event=None):
        """Toggles between Play and Pause states."""
        if self.is_playing:
            self.pause_audio()
        else:
            self.play_audio()

    def play_audio(self, from_loop=False):
        if self.is_playing and not from_loop:
            self.stop_audio()
            
        base_sample_rate = self.canvas_sample_rate
        speed = self.speed_slider.get()
        playback_sample_rate = int(base_sample_rate * speed)
        
        # 1. Background Engine Setup
        if not hasattr(self, 'cached_base_audio') or self.cached_base_audio is None or self.undo_redo_counter != 0:
            if DEBUG: print("Starting Background Audio Synthesis...")
            ft_data = self.generate_ft_data()
            
            # Pre-calculate the total array size mathematically
            step = ft_data.NFFT - ft_data.noverlap
            total_samples = int((len(ft_data.times) - 1) * step + ft_data.NFFT)
            
            self.cached_base_audio = np.zeros(total_samples, dtype=np.float32)
            self.cached_samples_ready = 0
            self.total_playback_samples = total_samples
            self.undo_redo_counter = 0
            
            self.synthesis_flag = time.time() # Create a unique token for the new thread
            threading.Thread(target=self.synthesize_audio_task, args=(ft_data, self.synthesis_flag), daemon=True).start()
            
            # Give the thread 0.2s to generate the first few chunks before we chase it with playback!
            self.root.after(200, lambda: self.start_playback_stream(playback_sample_rate))
        else:
            if DEBUG: print("Reading cached synthesized audio...")
            self.start_playback_stream(playback_sample_rate)

    def pause_audio(self):
        """Pauses audio and saves the exact playhead location to the slider."""
        self.stop_audio(pause=True)
        if DEBUG: print("Playback paused.")

    def stop_audio(self, pause=False):
        """Immediately halts audio and updates the slider to exact position."""
        self.is_playing = False
        if hasattr(self, 'playback_stream') and self.playback_stream:
            self.playback_stream.stop()
            self.playback_stream = None
            
        self.play_btn.config(text="▶ Play", bg="#4CAF50")
        
        if pause and hasattr(self, 'total_playback_samples') and self.total_playback_samples > 0:
            current_percent = self.playback_index / self.total_playback_samples
            self.time_slider.set(current_percent * 100)
            self.draw_playhead(int(current_percent * self.width))

        else:
            self.time_slider.set(0)
            self.draw_playhead(0)
            
        if DEBUG: print("Playback stopped.")

    def clear_audio_cache(self, *args):
        """Wipes the audio RAM cache so the next Play click forces a recalculation."""
        if self.prev_playback_source == self.playback_source.get():
            return
        self.prev_playback_source = self.playback_source.get()
        self.cached_base_audio = None
        if getattr(self, 'is_playing', False):
            self.stop_audio()

    def start_playback_stream(self, playback_sample_rate):
        start_percent = self.time_slider.get() / 100.0
        self.playback_index = int(self.total_playback_samples * start_percent)
        
        # Safety catch: auto-reset to beginning if playing from the very end
        if self.playback_index >= self.total_playback_samples * 0.99:
            self.playback_index = 0
            self.time_slider.set(0)
            
        self.is_playing = True
        self.play_btn.config(text="⏸ Pause", bg="#FF9800")
        
        # Fire up the C-Level OutputStream!
        self.playback_stream = sd.OutputStream(
            samplerate=playback_sample_rate, 
            channels=1, 
            callback=self.audio_callback,
            finished_callback=self.audio_finished_callback
        )
        self.playback_stream.start()
        if DEBUG: print("Streaming Playback Started...")
        
        self.update_playhead_loop()

    def synthesize_audio_task(self, ft_data, thread_token):
        """Background Thread: Streams generated audio chunks into the RAM cache silently."""
        for audio_chunk in AudioSignal.stream_from_ft_data(ft_data, columns_per_chunk=10):
            # If the user drew something and forced a new thread to start, kill this old ghost thread!
            if self.synthesis_flag != thread_token or getattr(self, 'cached_base_audio', None) is None: 
                break
                
            chunk_len = len(audio_chunk)

            if self.cached_samples_ready + chunk_len > len(self.cached_base_audio):
                break

            self.cached_base_audio[self.cached_samples_ready : self.cached_samples_ready + chunk_len] = audio_chunk
            self.cached_samples_ready += chunk_len
            
        if self.synthesis_flag == thread_token:
            if DEBUG: print("Background Synthesis Complete!")

    def audio_callback(self, outdata, frames, time_info, status):
        """Pulls audio dynamically from the background thread and feeds the speaker hardware."""
        valid_frames = min(frames, self.cached_samples_ready - self.playback_index)
        if valid_frames > 0:
            audio_chunk = self.cached_base_audio[self.playback_index : self.playback_index + valid_frames]
            
            outdata[:valid_frames, 0] = np.clip(audio_chunk * self.current_volume, -1.0, 1.0)
            
            if valid_frames < frames: outdata[valid_frames:, 0] = 0
            self.playback_index += valid_frames
        else:
            outdata.fill(0)
            if self.cached_samples_ready >= self.total_playback_samples:
                raise sd.CallbackStop()

    def audio_finished_callback(self):
        if not self.is_playing: return # Terminated by user stop button
        
        if self.loop_var.get():
            self.root.after(0, lambda: self.time_slider.set(0))
            self.root.after(0, lambda: self.play_audio(from_loop=True))
        else:
            self.root.after(0, self.stop_audio)

    def clear_overlay(self):
        """Clears the overlay matrix completely."""
        self.stop_audio()
        self.clear_audio_cache()
        
        if hasattr(self, 'overlay_matrix'):
            self.overlay_matrix.fill(0.0)
        if hasattr(self, 'overlay_phase_matrix'):
            self.overlay_phase_matrix.fill(0.5)
            
        for attr in ['pristine_overlay_amp_matrix', 'pristine_overlay_phase_matrix', 'overlay_phase_is_edited']:
            if hasattr(self, attr): delattr(self, attr)
            
        self.has_overlay = False
        self.render_matrix_to_image()

    def update_playhead_loop(self):
        if not self.is_playing: return
        
        if self.total_playback_samples > 0:
            current_percent = self.playback_index / self.total_playback_samples
            current_x = int(current_percent * self.width)
            self.draw_playhead(current_x)
            
        self.root.after(15, self.update_playhead_loop)

    def draw_playhead(self, x_pos, moved_time_slider=False):
        if self.is_playing and moved_time_slider:
            self.pause_audio()

        if self.playhead_triangle_id:
            self.canvas.delete(self.playhead_triangle_id)
        if self.playhead_line_id:
            self.canvas.delete(self.playhead_line_id)

        # Convert Matrix X to UI X
        x1, _, x2, _ = self.get_crop_bounds()
        ui_x_pos = ((x_pos - x1) / (x2 - x1)) * self.ui_width

        if not self.is_playing:
            ui_x_pos += 1.45
        else:
            ui_x_pos -= 3
                    
        if ui_x_pos < -20 or ui_x_pos > self.ui_width + 20:
            return

        size = 8
        self.playhead_triangle_id = self.canvas.create_polygon(
            round(ui_x_pos) - size, 0, round(ui_x_pos) + size, 0, round(ui_x_pos), size + 5,
            fill="yellow", outline="black"
        )
        if self.show_playhead_line.get():
            self.playhead_line_id = self.canvas.create_line(
                round(ui_x_pos), size + 5, round(ui_x_pos), round(self.ui_height),
                fill="yellow", dash=(4, 4)
            )

    def render_matrix_to_image(self):
        has_overlay = hasattr(self, 'has_overlay') and self.has_overlay
        show_overlay = getattr(self, 'show_overlay', tk.BooleanVar(value=False)).get()
        draw_in_red = getattr(self, 'draw_in_red', tk.BooleanVar(value=False)).get()

        x1, y1, x2, y2 = self.get_crop_bounds()

        # Check active view mode!
        viewing_phase = getattr(self, 'view_mode', tk.StringVar(value="Amplitude")).get() == "Phase"
        # Route the data arrays dynamically
        active_main = self.phase_matrix if viewing_phase else self.amp_matrix
        cropped_main = active_main[y1:y2, x1:x2]
        
        brightness = self.brightness_slider.get()
        
        # 1. Apply to main drawing
        if brightness != 0.0:
            disp_main = np.clip(cropped_main + brightness, 0.0, 1.0)
            full_main = np.clip(active_main + brightness, 0.0, 1.0)
        else:
            disp_main = cropped_main
            full_main = active_main
        
        if has_overlay and show_overlay:
            active_overlay = self.overlay_phase_matrix if viewing_phase else self.overlay_matrix
            cropped_overlay = active_overlay[y1:y2, x1:x2]
            
            if brightness != 0.0:
                disp_overlay = np.clip(cropped_overlay + brightness, 0.0, 1.0)
                full_overlay = np.clip(active_overlay + brightness, 0.0, 1.0)
            else:
                disp_overlay = cropped_overlay
                full_overlay = active_overlay
            
            if draw_in_red:
                overlay_colored = self.colormap(disp_overlay)[:, :, :3]
                alpha = disp_main[:, :, np.newaxis]
                red_layer = np.array([1.0, 0.0, 0.0])
                display_rgb = (red_layer * alpha) + (overlay_colored * (1.0 - alpha))
                
                full_overlay_colored = self.colormap(full_overlay)[:, :, :3]
                full_alpha = full_main[:, :, np.newaxis]
                full_rgb = (red_layer * full_alpha) + (full_overlay_colored * (1.0 - full_alpha))
                
            else:
                alpha_blend = self.overlay_opacity.get()
                display_slice = np.clip(disp_main + (disp_overlay * alpha_blend), 0.0, 1.0)
                full_slice = np.clip(full_main + (full_overlay * alpha_blend), 0.0, 1.0)
                
                display_rgb = self.colormap(display_slice)[:, :, :3]
                full_rgb = self.colormap(full_slice)[:, :, :3]
        else:
            display_rgb = self.colormap(disp_main)[:, :, :3]
            full_rgb = self.colormap(full_main)[:, :, :3]
            
        cropped_int = (display_rgb * 255).astype(np.uint8)
        full_int = (full_rgb * 255).astype(np.uint8)
        
        display_pil = Image.fromarray(cropped_int).resize((self.ui_width, self.ui_height), Image.Resampling.NEAREST)
        self.pil_image = Image.fromarray(full_int)
        
        self.tk_image = ImageTk.PhotoImage(image=display_pil)
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)

        if hasattr(self, 'show_grid'): self.update_grid()
        self.draw_axes()

    def swap_canvas_overlay(self):
        """Swaps the main canvas and the overlay matrices."""
        self.stop_audio()
        self.clear_audio_cache()
        self.save_state()
        
        # 1. Swap Amp
        temp_amp = self.amp_matrix.copy()
        self.amp_matrix = self.overlay_matrix.copy() if hasattr(self, 'overlay_matrix') else np.zeros((self.height, self.width), dtype=float)
        self.overlay_matrix = temp_amp
        
        # 2. Swap Phase
        temp_phase = self.phase_matrix.copy()
        self.phase_matrix = self.overlay_phase_matrix.copy() if hasattr(self, 'overlay_phase_matrix') else np.full((self.height, self.width), 0.5, dtype=float)
        self.overlay_phase_matrix = temp_phase
        
        # 3. Swap High-Def Pristine Matrices Safely!
        temp_pristine_amp = getattr(self, 'pristine_amp_matrix', None)
        if hasattr(self, 'pristine_overlay_amp_matrix'):
            self.pristine_amp_matrix = self.pristine_overlay_amp_matrix.copy()
        elif hasattr(self, 'pristine_amp_matrix'):
            delattr(self, 'pristine_amp_matrix')
        if temp_pristine_amp is not None:
            self.pristine_overlay_amp_matrix = temp_pristine_amp
        elif hasattr(self, 'pristine_overlay_amp_matrix'):
            delattr(self, 'pristine_overlay_amp_matrix')

        temp_pristine_phase = getattr(self, 'pristine_phase_matrix', None)
        if hasattr(self, 'pristine_overlay_phase_matrix'):
            self.pristine_phase_matrix = self.pristine_overlay_phase_matrix.copy()
        elif hasattr(self, 'pristine_phase_matrix'):
            delattr(self, 'pristine_phase_matrix')
        if temp_pristine_phase is not None:
            self.pristine_overlay_phase_matrix = temp_pristine_phase
        elif hasattr(self, 'pristine_overlay_phase_matrix'):
            delattr(self, 'pristine_overlay_phase_matrix')
            
        # 4. Swap the squashed base & Phase Edit Flags!
        if hasattr(self, 'base_squashed_amp'):
            delattr(self, 'base_squashed_amp')
        self.base_squashed_amp = self.amp_matrix.copy()
        
        temp_edited = getattr(self, 'phase_is_edited', False)
        self.phase_is_edited = getattr(self, 'overlay_phase_is_edited', False)
        self.overlay_phase_is_edited = temp_edited
        
        self.has_overlay = True
        self.render_matrix_to_image()

    def start_stroke(self, event):
        if self.undo_redo_counter < 0: self.undo_redo_counter = 0
        self.undo_redo_counter += 1
        self.stop_audio()
        super().start_stroke(event)

    def undo(self, event=None):
        self.undo_redo_counter -= 1
        self.stop_audio()
        super().undo(event)
    
    def redo(self, event=None):
        self.undo_redo_counter += 1
        self.stop_audio()
        super().redo(event)

    def generate_ft_data(self):
        """Synthesizes Audio using Complex Vector Superposition."""
        duration_seconds = getattr(self, 'canvas_duration_seconds', 5.0)
        sample_rate = getattr(self, 'canvas_sample_rate', 44100)
        NFFT = (self.height - 1) * 2 
        ideal_step = NFFT // 2
        total_samples = int(sample_rate * duration_seconds)
        required_width = max(2, total_samples // ideal_step)

        is_editing_imported = hasattr(self, 'pristine_amp_matrix') and getattr(self.import_mode, 'get', lambda: "edit")() == "edit"
        has_overlay_matrix = hasattr(self, 'has_overlay') and self.has_overlay
        
        if is_editing_imported:
            required_width = self.pristine_amp_matrix.shape[1]
            
        src = getattr(self, 'playback_source', tk.StringVar(value="Both (Mix)")).get()
        max_val = getattr(self, 'original_max_val', float(self.height - 1))

        times = np.arange(required_width) * (ideal_step / sample_rate)
        freqs = np.linspace(0, sample_rate / 2, self.height)
        phase_advance = 2 * np.pi * freqs[:, np.newaxis] * times[np.newaxis, :]

        # --- 1. EXTRACT MAIN CANVAS COMPLEX SIGNAL ---
        if is_editing_imported:
            ui_delta = self.amp_matrix - getattr(self, 'base_squashed_amp')
            if ui_delta.shape[1] != required_width:
                pil_delta = Image.fromarray(ui_delta.astype(np.float32), mode="F")
                pil_delta = pil_delta.resize((required_width, self.height), Image.Resampling.BILINEAR)
                high_def_delta = np.array(pil_delta)
            else:
                high_def_delta = ui_delta
                
            main_amp = self.pristine_amp_matrix + high_def_delta
            main_amp = np.clip(main_amp, 0.0, 1.0)
            main_amp = np.flipud(main_amp) * max_val
            
            if hasattr(self, 'pristine_phase_matrix') and not getattr(self, 'phase_is_edited', False):
                main_phase = np.flipud(self.pristine_phase_matrix)
            else:
                ui_phase = (np.flipud(self.phase_matrix) * 2 * np.pi) - np.pi
                if ui_phase.shape[1] != required_width:
                    Z_ui = np.exp(1j * ui_phase)
                    real_pil = Image.fromarray(np.real(Z_ui).astype(np.float32), mode="F")
                    imag_pil = Image.fromarray(np.imag(Z_ui).astype(np.float32), mode="F")
                    real_res = np.array(real_pil.resize((required_width, self.height), Image.Resampling.BILINEAR))
                    imag_res = np.array(imag_pil.resize((required_width, self.height), Image.Resampling.BILINEAR))
                    main_phase = np.angle(real_res + 1j * imag_res) + phase_advance
                else:
                    main_phase = ui_phase + phase_advance
        else:
            active_amp = np.flipud(self.amp_matrix)
            if active_amp.shape[1] != required_width:
                pil_amp = Image.fromarray(active_amp.astype(np.float32), mode="F")
                pil_amp = pil_amp.resize((required_width, self.height), Image.Resampling.BILINEAR)
                main_amp = np.array(pil_amp) * max_val
            else:
                main_amp = active_amp * max_val
                
            ui_phase = (np.flipud(self.phase_matrix) * 2 * np.pi) - np.pi
            if ui_phase.shape[1] != required_width:
                Z_ui = np.exp(1j * ui_phase)
                real_pil = Image.fromarray(np.real(Z_ui).astype(np.float32), mode="F")
                imag_pil = Image.fromarray(np.imag(Z_ui).astype(np.float32), mode="F")
                real_res = np.array(real_pil.resize((required_width, self.height), Image.Resampling.BILINEAR))
                imag_res = np.array(imag_pil.resize((required_width, self.height), Image.Resampling.BILINEAR))
                main_phase = np.angle(real_res + 1j * imag_res) + phase_advance
            else:
                main_phase = ui_phase + phase_advance

        # --- 2. EXTRACT OVERLAY COMPLEX SIGNAL ---
        overlay_amp = np.zeros((self.height, required_width))
        overlay_phase = phase_advance.copy()
        
        if has_overlay_matrix:
            if hasattr(self, 'pristine_overlay_amp_matrix') and not getattr(self, 'overlay_phase_is_edited', False):
                hd_ov_amp = self.pristine_overlay_amp_matrix
                if hd_ov_amp.shape[1] != required_width:
                    pil_ov = Image.fromarray(hd_ov_amp.astype(np.float32), mode="F")
                    pil_ov = pil_ov.resize((required_width, self.height), Image.Resampling.BILINEAR)
                    overlay_amp = np.flipud(np.array(pil_ov)) * max_val
                else:
                    overlay_amp = np.flipud(hd_ov_amp) * max_val
                    
                hd_ov_phase = np.flipud(self.pristine_overlay_phase_matrix)
                if hd_ov_phase.shape[1] != required_width:
                    Z_ov = np.exp(1j * hd_ov_phase)
                    real_pil = Image.fromarray(np.real(Z_ov).astype(np.float32), mode="F")
                    imag_pil = Image.fromarray(np.imag(Z_ov).astype(np.float32), mode="F")
                    real_res = np.array(real_pil.resize((required_width, self.height), Image.Resampling.BILINEAR))
                    imag_res = np.array(imag_pil.resize((required_width, self.height), Image.Resampling.BILINEAR))
                    overlay_phase = np.angle(real_res + 1j * imag_res)
                else:
                    overlay_phase = hd_ov_phase
            else:
                ui_ov = np.flipud(self.overlay_matrix)
                if ui_ov.shape[1] != required_width:
                    pil_ov = Image.fromarray(ui_ov.astype(np.float32), mode="F")
                    pil_ov = pil_ov.resize((required_width, self.height), Image.Resampling.BILINEAR)
                    overlay_amp = np.array(pil_ov) * max_val
                else:
                    overlay_amp = ui_ov * max_val
                    
                ui_ph = (np.flipud(self.overlay_phase_matrix) * 2 * np.pi) - np.pi
                if ui_ph.shape[1] != required_width:
                    Z_ui = np.exp(1j * ui_ph)
                    real_pil = Image.fromarray(np.real(Z_ui).astype(np.float32), mode="F")
                    imag_pil = Image.fromarray(np.imag(Z_ui).astype(np.float32), mode="F")
                    real_res = np.array(real_pil.resize((required_width, self.height), Image.Resampling.BILINEAR))
                    imag_res = np.array(imag_pil.resize((required_width, self.height), Image.Resampling.BILINEAR))
                    overlay_phase = np.angle(real_res + 1j * imag_res) + phase_advance
                else:
                    overlay_phase = ui_ph + phase_advance

        # --- 3. MULTIPLEXER (COMPLEX SUPERPOSITION) ---
        if src == "Overlay / Original":
            if is_editing_imported:
                audio_matrix = np.flipud(self.pristine_amp_matrix) * max_val
                final_phase_matrix = np.flipud(self.pristine_phase_matrix)
            else:
                audio_matrix = overlay_amp
                final_phase_matrix = overlay_phase
                
        elif src == "Main Canvas":
            audio_matrix = main_amp
            final_phase_matrix = main_phase
            
        else: # Both (Mix)
            Z_main = main_amp * np.exp(1j * main_phase)
            Z_overlay = overlay_amp * np.exp(1j * overlay_phase)
            Z_mix = Z_main + Z_overlay
            
            audio_matrix = np.abs(Z_mix)
            final_phase_matrix = np.angle(Z_mix)

        return FourierTransformData(
            amp_matrix=audio_matrix, phase_matrix=final_phase_matrix,
            freqs=freqs, times=times, sample_rate=sample_rate,
            NFFT=NFFT, noverlap=max(0, NFFT - ideal_step), useHanning=True
        )

    def apply_phase_generator(self):
        """Mathematically generates a phase mapping across the UI."""
        self.stop_audio()
        self.clear_audio_cache()
        self.save_state()
        mode = self.phase_gen_mode.get()
        self.phase_is_edited = True
        
        if mode == "0 Phase":
            self.phase_matrix.fill(0.5)
        elif mode == "Random Phase":
            self.phase_matrix = np.random.uniform(0.0, 1.0, size=self.phase_matrix.shape)
        elif mode == "Saved Phase" and hasattr(self, 'pristine_phase_matrix'):
            sr = getattr(self, 'canvas_sample_rate', 44100)
            ideal_step = ((self.height - 1) * 2) // 2
            required_width = self.pristine_phase_matrix.shape[1]
            
            times = np.arange(required_width) * (ideal_step / sr)
            freqs = np.linspace(0, sr / 2, self.height)
            
            phase_advance = 2 * np.pi * freqs[:, np.newaxis] * times[np.newaxis, :]
            
            # Flip the math so 0 Hz matches the Pristine UI orientation!
            phase_advance_ui = np.flipud(phase_advance)
            
            pure_offset = self.pristine_phase_matrix - phase_advance_ui
            pure_offset = (pure_offset + np.pi) % (2 * np.pi) - np.pi
            
            normalized_phase = (pure_offset + np.pi) / (2 * np.pi)
            
            pil_phase = Image.fromarray(normalized_phase.astype(np.float32), mode="F")
            pil_phase = pil_phase.resize((self.width, self.height), Image.Resampling.NEAREST)
            self.phase_matrix = np.array(pil_phase)
            self.phase_is_edited = False
        else: # Semi-Random
            # --- Semi-Random is now just scaled random offsets! ---
            semi_offset = np.random.uniform(-np.pi, np.pi, size=self.phase_matrix.shape) * 0.5
            self.phase_matrix = (semi_offset + np.pi) / (2 * np.pi)

        self.render_matrix_to_image()

    def clear_canvas(self):
        self.stop_audio()
        self.undo_redo_counter += 1
        super().clear_canvas()

    def save_file(self, event=None):
        """Exports the canvas to various formats, bypassing directory locks."""
        self.stop_audio()

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
            signal = ft_data.to_AudioSignal()
            
            # --- Gentle Peak Limiter to prevent clipping ---
            max_amp = np.max(np.abs(signal.time_vs_amplitude["Amplitude"].values))
            if max_amp > 1.0:
                signal.time_vs_amplitude["Amplitude"] /= max_amp

            signal.apply_volume_change(self.volume_slider.get())
                
            signal.to_audio_file(filepath, use_absolute_path=True)
            
        elif ext == "npz":
            filepath = filepath[:-4] + f" - Width-{self.width}, Height-{self.height}.npz"
            ft_data.to_npz_file(filepath, use_absolute_path=True)
            
        elif ext == "csv":
            signal = ft_data.to_AudioSignal()
            
            max_amp = np.max(np.abs(signal.time_vs_amplitude["Amplitude"].values))
            if max_amp > 1.0:
                signal.time_vs_amplitude["Amplitude"] /= max_amp
                
            signal.apply_volume_change(self.volume_slider.get())
                
            signal.to_csv_file(filepath, use_absolute_path=True)
            
        mb.showinfo("Saved", f"Data saved successfully to {filepath}")

    def open_file(self, event=None):
        """Imports data from any location, resizing it to perfectly fit the canvas."""
        self.stop_audio()

        filepath = fd.askopenfilename(
            title="Select data file",
            initialdir=os.getcwd(),
            filetypes=[("Audio/Fourier Transform data/Image", "*.wav *.mp3 *.ogg *.csv *.npz *.png *.jpg *.jpeg")]
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

            if not create_cache_data and ext != "png":
                ft_data = FourierTransformData.from_npz_file(cached_file_path, use_absolute_path=True)
                # true_duration = AudioSignal.from_ft_data(ft_data).get_audio_duration()

            elif ext in ["png", "jpg", "jpeg"]:
                create_cache_data = False
                
                # Convert straight to Greyscale Amplitude
                img = Image.open(filepath).convert("L") 
                img = img.resize((self.width, self.height), Image.Resampling.BILINEAR)
                
                final_amp_matrix = np.array(img).astype(float) / 255.0
                self.original_max_val = float(NFFT / 2)
                
                # PNGs don't have audio waves, so manually generate a scaled random phase!
                if getattr(self, "random_phase_matrix", None) is not None:
                    imported_phase = self.random_phase_matrix
                else:
                    imported_phase = np.random.uniform(0.0, 1.0, size=(self.height, self.width))
                final_phase_matrix = imported_phase.copy()

            # 1. Extract the raw FT Data using raw library methods to bypass directory locks
            elif ext in ["wav", "mp3", "ogg"]:
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
                    raise ValueError("Unrecognized CSV format. Cannot determine if Audio or Fourier Transform data.")
                    
            elif ext == "npz":
                if not filepath[:-4].split(" - ")[-1] == f"Width-{self.width}, Height-{self.height}":
                    mb.showerror("Import Error", f"Canvas height ({self.height}) does not match file height ({filepath[:-4].split(" - ")[-1].split(", ")[-1][len("Height-"):]})")
                    return
                ft_data = FourierTransformData.from_npz_file(filepath, use_absolute_path=True)
                create_cache_data = False
                # true_duration = AudioSignal.from_ft_data(ft_data).get_audio_duration()

            # 2. Normalize and Prepare the Matrix
            if ext not in ["png", "jpg", "jpeg"]:
                self.canvas_sample_rate = int(ft_data.sample_rate)
                if true_duration:
                    self.canvas_duration_seconds = float(true_duration)
                elif len(ft_data.times) > 1:
                    self.canvas_duration_seconds = float(ft_data.times[-1] - ft_data.times[0])

                imported_amp = ft_data.amp_matrix
                
                self.original_max_val = np.max(imported_amp)
                if self.original_max_val > 0:
                    imported_amp = imported_amp / self.original_max_val
                    
                imported_amp = np.flipud(imported_amp)
                
                # --- FIX 4: Strip Carrier Advance to map STFT Phase to UI Offsets ---
                imported_phase_raw = np.flipud(ft_data.phase_matrix)
                
                required_width = imported_phase_raw.shape[1]
                times = np.arange(required_width) * (ideal_step / self.canvas_sample_rate)
                freqs = np.linspace(0, self.canvas_sample_rate / 2, self.height)
                
                phase_advance = 2 * np.pi * freqs[:, np.newaxis] * times[np.newaxis, :]
                phase_advance_ui = np.flipud(phase_advance)
                
                pure_offset = imported_phase_raw - phase_advance_ui
                pure_offset = (pure_offset + np.pi) % (2 * np.pi) - np.pi
                imported_phase = (pure_offset + np.pi) / (2 * np.pi)

                # --- FIX 1: Squash for UI using 32-bit floats to prevent Noise Gating! ---
                pil_img = Image.fromarray(imported_amp.astype(np.float32), mode="F")
                pil_img = pil_img.resize((self.width, self.height), Image.Resampling.BILINEAR)
                final_amp_matrix = np.array(pil_img)
                
                # Squash safely for Phase UI (Must use NEAREST)
                pil_phase = Image.fromarray(imported_phase.astype(np.float32), mode="F")
                pil_phase = pil_phase.resize((self.width, self.height), Image.Resampling.NEAREST)
                final_phase_matrix = np.array(pil_phase)
            
            # 4. Apply based on user's mode preference
            if self.import_mode.get() == "edit":
                self.save_state()
                self.phase_is_edited = False if ext not in ["png", "jpg", "jpeg"] else True
                self.amp_matrix = final_amp_matrix.copy()
                self.phase_matrix = final_phase_matrix.copy()
                
                self.base_squashed_amp = final_amp_matrix.copy()
                if ext not in ["png", "jpg", "jpeg"]:
                    self.pristine_amp_matrix = imported_amp.copy()
                    self.pristine_phase_matrix = imported_phase_raw.copy()
            elif self.import_mode.get() == "overlay":
                self.overlay_matrix = final_amp_matrix
                self.overlay_phase_matrix = final_phase_matrix.copy()

                if ext not in ["png", "jpg", "jpeg"]:
                    self.pristine_overlay_amp_matrix = imported_amp.copy()
                    self.pristine_overlay_phase_matrix = imported_phase_raw.copy()

                self.has_overlay = True

            self.cached_base_audio = None

            self.time_slider.set(0)
            self.draw_playhead(0)
                
            self.render_matrix_to_image()

            if create_cache_data and ext != "png":
                ft_data = self.generate_ft_data()
                ft_data.to_npz_file(cached_file_path, use_absolute_path=True)
            
        except Exception as e:
            mb.showerror("Import Error", f"Failed to load file:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    # app = DrawingProgram(root)
    app = SpectrogramSynthesizer(root, ui_width=2000, ui_height=1000, res_width=2000, res_height =513)
    root.mainloop()
# fir_window.py

import os
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np

from dsp import (
    PhaseBand,
    MagBand,
    build_phase_target,
    build_magnitude_target,
    synthesize_impulse_from_Hhalf,
)

# ==== Defaults you can tweak easily ====
DEFAULT_FIR_GEOMETRY = (1200, 600)   # width, height when the window opens
DEFAULT_FIR_MINSIZE  = (1200, 600)   # minimum allowed size


class FIRWindow(tk.Toplevel):
    """
    FIR editor window (per-trace) with:
      • Magnitude EQ (linear-phase) and Phase EQ tabs
      • Live preview to main plots (capped or full-res)
      • CSV export of the synthesized FIR impulse (exact N=taps samples)
      • Optional custom initial size and minimum size

    Usage:
        FIRWindow(master, 'A')                         # uses defaults
        FIRWindow(master, 'B', geometry=(1300,800))    # custom size
        FIRWindow(master, 'A', minsize=(1100,720))     # custom min size
        FIRWindow(master, 'B', geometry=(1300,800), minsize=(1100,720))
    """

    def __init__(self, master, which, *, geometry=DEFAULT_FIR_GEOMETRY, minsize=DEFAULT_FIR_MINSIZE):
        super().__init__(master)
        self.master = master
        self.which = which

        # ---- Window chrome / sizing ----
        self.title(f"FIR {which} — Linear-Phase Magnitude EQ + Phase EQ")
        try:
            w, h = geometry
            self.geometry(f"{int(w)}x{int(h)}")
        except Exception:
            self.geometry(f"{DEFAULT_FIR_GEOMETRY[0]}x{DEFAULT_FIR_GEOMETRY[1]}")
        try:
            mw, mh = minsize
            self.minsize(int(mw), int(mh))
        except Exception:
            self.minsize(DEFAULT_FIR_MINSIZE[0], DEFAULT_FIR_MINSIZE[1])
        self.resizable(True, True)

        # ---- Settings / state ----
        self.fs_var = tk.StringVar(value="48000")   # Hz
        self.taps_var = tk.StringVar(value="16384")
        self.fft_var = tk.StringVar(value="65536")
        self.auto_fft_var = tk.BooleanVar(value=True)
        self.fft_mult_var = tk.StringVar(value="x16")        # x4/x8/x16
        self.preview_cap_var = tk.StringVar(value="131072")  # live preview FFT cap
        self.hi_res_preview_var = tk.BooleanVar(value=False) # preview uses full auto-FFT when True
        self.centering_var = tk.StringVar(value="middle")    # start/middle/closest/custom
        self.custom_shift_var = tk.StringVar(value="0")
        self.window_var = tk.StringVar(value="hann")         # none/hann/blackman
        self.filename_var = tk.StringVar(value=f"impulse_{which.lower()}")
        self.dir_var = tk.StringVar(value=os.path.expanduser("~"))
        self.live_preview = tk.BooleanVar(value=True)

        # slider ranges & bands
        self.phase_range_var = tk.StringVar(value="±90°")
        self.mag_range_var   = tk.StringVar(value="±12 dB")
        self.phase_bands = [PhaseBand(f, Q=2.0, gain_deg=0.0, bypass=False) for f in np.geomspace(20.0, 20000.0, 10)]
        self.mag_bands   = [MagBand(  f, Q=2.0, gain_db =0.0, bypass=False) for f in np.geomspace(20.0, 20000.0, 10)]

        # Keep widget references for slider rebuilds
        self._phase_widgets = []
        self._mag_widgets = []

        # Build UI and kick off preview after initial draw
        self._build_ui()

        # If main holds a stored snapshot for this trace, apply it
        try:
            snapshot = self.master._firA_defs if which == 'A' else self.master._firB_defs
            if snapshot:
                self._apply_snapshot_defs(snapshot)
        except Exception:
            pass

        self.after(60, self._push_preview)

    # ---------------- UI ----------------
    def _build_ui(self):
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        nb = ttk.Notebook(left)
        self.tab_mag = ttk.Frame(nb)
        self.tab_phase = ttk.Frame(nb)
        nb.add(self.tab_mag, text="Magnitude EQ (linear-phase)")
        nb.add(self.tab_phase, text="Phase EQ")
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Magnitude tab ---
        top_mag = ttk.Frame(self.tab_mag); top_mag.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top_mag, text="Range:").pack(side=tk.LEFT)
        mag_rng = ttk.Combobox(top_mag, values=["±6 dB", "±12 dB", "±24 dB"], width=8,
                               state="readonly", textvariable=self.mag_range_var)
        mag_rng.pack(side=tk.LEFT, padx=6); mag_rng.bind("<<ComboboxSelected>>", lambda e: self._update_mag_slider_ranges())
        ttk.Checkbutton(top_mag, text="Live preview", variable=self.live_preview,
                        command=self._maybe_preview).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_mag, text="Add Band", command=self._add_mag_band).pack(side=tk.LEFT, padx=6)
        ttk.Button(top_mag, text="Remove Last", command=self._remove_last_mag_band).pack(side=tk.LEFT, padx=6)
        ttk.Button(top_mag, text="Reset gains", command=lambda: (self._reset_mag_gains(), self._maybe_preview())).pack(side=tk.LEFT, padx=6)

        self.mag_canvas = tk.Canvas(self.tab_mag, borderwidth=0, height=460)
        self.mag_sliders = ttk.Frame(self.mag_canvas)
        mag_hsb = ttk.Scrollbar(self.tab_mag, orient="horizontal", command=self.mag_canvas.xview)
        self.mag_canvas.configure(xscrollcommand=mag_hsb.set)
        mag_hsb.pack(side="bottom", fill="x")
        self.mag_canvas.pack(side="top", fill="both", expand=True)
        self.mag_canvas.create_window((0, 0), window=self.mag_sliders, anchor="nw")
        self.mag_sliders.bind("<Configure>", lambda e: self.mag_canvas.configure(scrollregion=self.mag_canvas.bbox("all")))

        # --- Phase tab ---
        top_ph = ttk.Frame(self.tab_phase); top_ph.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top_ph, text="Range:").pack(side=tk.LEFT)
        ph_rng = ttk.Combobox(top_ph, values=["±45°", "±90°", "±180°"], width=7,
                              state="readonly", textvariable=self.phase_range_var)
        ph_rng.pack(side=tk.LEFT, padx=6); ph_rng.bind("<<ComboboxSelected>>", lambda e: self._update_phase_slider_ranges())
        ttk.Checkbutton(top_ph, text="Live preview", variable=self.live_preview,
                        command=self._maybe_preview).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_ph, text="Add Band", command=self._add_phase_band).pack(side=tk.LEFT, padx=6)
        ttk.Button(top_ph, text="Remove Last", command=self._remove_last_phase_band).pack(side=tk.LEFT, padx=6)
        ttk.Button(top_ph, text="Reset gains", command=lambda: (self._reset_phase_gains(), self._maybe_preview())).pack(side=tk.LEFT, padx=6)

        self.phase_canvas = tk.Canvas(self.tab_phase, borderwidth=0, height=460)
        self.phase_sliders = ttk.Frame(self.phase_canvas)
        phase_hsb = ttk.Scrollbar(self.tab_phase, orient="horizontal", command=self.phase_canvas.xview)
        self.phase_canvas.configure(xscrollcommand=phase_hsb.set)
        phase_hsb.pack(side="bottom", fill="x")
        self.phase_canvas.pack(side="top", fill="both", expand=True)
        self.phase_canvas.create_window((0, 0), window=self.phase_sliders, anchor="nw")
        self.phase_sliders.bind("<Configure>", lambda e: self.phase_canvas.configure(scrollregion=self.phase_canvas.bbox("all")))

        # Build initial slider columns
        self._rebuild_mag_sliders(); self._rebuild_phase_sliders()
        self._update_mag_slider_ranges(); self._update_phase_slider_ranges()

        # ---- Right column (settings / export) ----
        right = ttk.Frame(self); right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        g = ttk.LabelFrame(right, text="Impulse Settings"); g.pack(fill=tk.X, pady=4)
        row = ttk.Frame(g); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="taps").grid(row=0, column=0, sticky="e")
        taps_entry = ttk.Entry(row, width=9, textvariable=self.taps_var); taps_entry.grid(row=0, column=1, padx=4)

        ttk.Label(row, text="FFT length").grid(row=1, column=0, sticky="e")
        fft_entry = ttk.Entry(row, width=9, textvariable=self.fft_var); fft_entry.grid(row=1, column=1, padx=4)
        ttk.Checkbutton(row, text="Auto FFT length", variable=self.auto_fft_var, command=self._maybe_auto_fft).grid(row=1, column=2, padx=(10, 6))
        ttk.Label(row, text="mult").grid(row=1, column=3, sticky="e")
        mult_cb = ttk.Combobox(row, values=["x4", "x8", "x16"], width=4, state="readonly", textvariable=self.fft_mult_var)
        mult_cb.grid(row=1, column=4, padx=(4, 0)); mult_cb.bind("<<ComboboxSelected>>", lambda e: self._maybe_auto_fft())

        p_row = ttk.Frame(g); p_row.pack(fill=tk.X, pady=2)
        ttk.Label(p_row, text="Preview FFT cap").grid(row=0, column=0, sticky="e")
        cap_cb = ttk.Combobox(p_row, values=["65536", "131072", "262144"], width=8, state="readonly", textvariable=self.preview_cap_var)
        cap_cb.grid(row=0, column=1, padx=4); cap_cb.bind("<<ComboboxSelected>>", lambda e: self._maybe_preview())
        ttk.Checkbutton(p_row, text="Hi-res preview (use full auto-FFT)",
                        variable=self.hi_res_preview_var, command=self._maybe_preview).grid(row=0, column=2, padx=8)

        ttk.Label(row, text="centering").grid(row=2, column=0, sticky="e")
        cbox = ttk.Combobox(row, values=["start", "middle", "closest", "custom"], width=10,
                            state="readonly", textvariable=self.centering_var)
        cbox.grid(row=2, column=1, padx=4)
        ttk.Label(row, text="custom shift (samples)").grid(row=3, column=0, sticky="e")
        ttk.Entry(row, width=9, textvariable=self.custom_shift_var).grid(row=3, column=1, padx=4)

        ttk.Label(row, text="windowing").grid(row=4, column=0, sticky="e")
        wbox = ttk.Combobox(row, values=["none", "hann", "blackman"], width=8,
                            state="readonly", textvariable=self.window_var)
        wbox.grid(row=4, column=1, padx=4)

        ttk.Label(row, text="rate (Hz)").grid(row=5, column=0, sticky="e")
        ttk.Entry(row, width=9, textvariable=self.fs_var).grid(row=5, column=1, padx=4)

        ttk.Label(row, text="filename").grid(row=6, column=0, sticky="e")
        ttk.Entry(row, width=16, textvariable=self.filename_var).grid(row=6, column=1, padx=4)

        pathf = ttk.Frame(g); pathf.pack(fill=tk.X, pady=2)
        ttk.Label(pathf, text="directory").grid(row=0, column=0, sticky="e")
        ttk.Entry(pathf, width=28, textvariable=self.dir_var).grid(row=0, column=1, padx=4)
        ttk.Button(pathf, text="Browse…", command=self._pick_dir).grid(row=0, column=2, padx=4)

        ttk.Button(right, text=f"Generate {self.which} (.csv)", command=self._generate_csv).pack(fill=tk.X, pady=8)
        ttk.Label(right, text="Export includes BOTH: linear-phase Magnitude EQ and Phase EQ.",
                  foreground="#555").pack(fill=tk.X, pady=(2, 8))
        self.status = ttk.Label(right, text=f"Ready. Live preview on for {self.which}.", foreground="#2a2")
        self.status.pack(fill=tk.X)

        taps_entry.bind("<Return>", lambda e: (self._maybe_auto_fft(), self._maybe_preview()))
        taps_entry.bind("<FocusOut>", lambda e: (self._maybe_auto_fft(), self._maybe_preview()))
        fft_entry.bind("<Return>", lambda e: self._maybe_preview())
        fft_entry.bind("<FocusOut>", lambda e: self._maybe_preview())

    # --------- Auto FFT length ---------
    def _pow2ceil(self, n):
        n = int(max(1, n))
        return 1 << (n - 1).bit_length()

    def _maybe_auto_fft(self):
        if not self.auto_fft_var.get():
            return
        try:
            taps = int(float(self.taps_var.get()))
        except Exception:
            return
        mult = {"x4": 4, "x8": 8, "x16": 16}.get(self.fft_mult_var.get(), 16)
        target = self._pow2ceil(max(mult * max(1, taps), 8192))  # clamp min 8192
        target = min(target, 1_048_576)
        self.fft_var.set(str(target))
        self._maybe_preview()

    # --------- Sliders: Magnitude ---------
    def _rebuild_mag_sliders(self):
        for child in self.mag_sliders.winfo_children():
            child.destroy()
        self._mag_widgets.clear()
        for i, b in enumerate(self.mag_bands):
            col = ttk.Frame(self.mag_sliders, padding=(4, 2))
            col.grid(row=0, column=i, sticky="n")
            s = tk.Scale(col, from_=12, to=-12, resolution=0.1, length=300, orient="vertical",
                         width=10, sliderlength=12, command=lambda val, idx=i: self._on_mag_gain_changed(idx, val))
            s.set(b.gain_db); s.pack(side=tk.TOP)
            ttk.Label(col, text="+dB\n\n0\n\n-dB", justify="center").pack(side=tk.TOP, pady=(2, 4))
            hz_var = tk.StringVar(value=f"{b.f_hz:.2f}")
            q_var  = tk.StringVar(value=f"{b.Q:.2f}")
            by_var = tk.BooleanVar(value=b.bypass)
            hz_box = ttk.Entry(col, width=8, textvariable=hz_var, justify="center"); hz_box.pack(pady=(2, 0))
            ttk.Label(col, text="Hz").pack()
            q_box  = ttk.Entry(col, width=8, textvariable=q_var, justify="center"); q_box.pack(pady=(2, 0))
            ttk.Label(col, text="Q").pack()
            ttk.Checkbutton(col, text="bypass", variable=by_var,
                            command=lambda idx=i, v=by_var: self._on_mag_bypass(idx, v)).pack(pady=(4, 2))
            hz_box.bind("<FocusOut>", lambda e, idx=i, var=hz_var: self._on_mag_freq_changed(idx, var))
            hz_box.bind("<Return>",   lambda e, idx=i, var=hz_var: self._on_mag_freq_changed(idx, var))
            q_box.bind("<FocusOut>",  lambda e, idx=i, var=q_var:  self._on_mag_q_changed(idx, var))
            q_box.bind("<Return>",    lambda e, idx=i, var=q_var:  self._on_mag_q_changed(idx, var))
            self._mag_widgets.append((s, hz_var, q_var, by_var))

    def _mag_range_value(self):
        txt = self.mag_range_var.get()
        if "6" in txt: return 6.0
        if "24" in txt: return 24.0
        return 12.0

    def _update_mag_slider_ranges(self):
        r = self._mag_range_value()
        for (s, _, _, _) in self._mag_widgets:
            s.configure(from_=r, to=-r)

    def _add_mag_band(self):
        freqs = [b.f_hz for b in self.mag_bands] or [20.0, 20000.0]
        f_new = np.sqrt(min(freqs) * max(freqs))
        self.mag_bands.append(MagBand(f_new, 2.0, 0.0, False))
        self._rebuild_mag_sliders(); self._update_mag_slider_ranges(); self._maybe_preview()

    def _remove_last_mag_band(self):
        if self.mag_bands:
            self.mag_bands.pop()
            self._rebuild_mag_sliders(); self._update_mag_slider_ranges(); self._maybe_preview()

    def _reset_mag_gains(self):
        for b in self.mag_bands:
            b.gain_db = 0.0
        for (s, _, _, _) in self._mag_widgets:
            s.set(0)

    def _on_mag_gain_changed(self, idx, val):
        try:
            self.mag_bands[idx].gain_db = float(val)
        except Exception:
            pass
        self._maybe_preview()

    def _on_mag_freq_changed(self, idx, var):
        try:
            self.mag_bands[idx].f_hz = max(1.0, float(var.get()))
        except Exception:
            pass
        self._maybe_preview()

    def _on_mag_q_changed(self, idx, var):
        try:
            self.mag_bands[idx].Q = max(0.05, float(var.get()))
        except Exception:
            pass
        self._maybe_preview()

    def _on_mag_bypass(self, idx, by_var):
        self.mag_bands[idx].bypass = bool(by_var.get())
        self._maybe_preview()

    # --------- Sliders: Phase ---------
    def _rebuild_phase_sliders(self):
        for child in self.phase_sliders.winfo_children():
            child.destroy()
        self._phase_widgets.clear()
        for i, b in enumerate(self.phase_bands):
            col = ttk.Frame(self.phase_sliders, padding=(4, 2))
            col.grid(row=0, column=i, sticky="n")
            s = tk.Scale(col, from_=90, to=-90, resolution=1, length=300, orient="vertical",
                         width=10, sliderlength=12, command=lambda val, idx=i: self._on_phase_gain_changed(idx, val))
            s.set(b.gain_deg); s.pack(side=tk.TOP)
            ttk.Label(col, text="+°\n\n0\n\n-°", justify="center").pack(side=tk.TOP, pady=(2, 4))
            hz_var = tk.StringVar(value=f"{b.f_hz:.2f}")
            q_var  = tk.StringVar(value=f"{b.Q:.2f}")
            by_var = tk.BooleanVar(value=b.bypass)
            hz_box = ttk.Entry(col, width=8, textvariable=hz_var, justify="center"); hz_box.pack(pady=(2, 0))
            ttk.Label(col, text="Hz").pack()
            q_box  = ttk.Entry(col, width=8, textvariable=q_var, justify="center"); q_box.pack(pady=(2, 0))
            ttk.Label(col, text="Q").pack()
            ttk.Checkbutton(col, text="bypass", variable=by_var,
                            command=lambda idx=i, v=by_var: self._on_phase_bypass(idx, v)).pack(pady=(4, 2))
            hz_box.bind("<FocusOut>", lambda e, idx=i, var=hz_var: self._on_phase_freq_changed(idx, var))
            hz_box.bind("<Return>",   lambda e, idx=i, var=hz_var: self._on_phase_freq_changed(idx, var))
            q_box.bind("<FocusOut>",  lambda e, idx=i, var=q_var:  self._on_phase_q_changed(idx, var))
            q_box.bind("<Return>",    lambda e, idx=i, var=q_var:  self._on_phase_q_changed(idx, var))
            self._phase_widgets.append((s, hz_var, q_var, by_var))

    def _phase_range_value(self):
        txt = self.phase_range_var.get()
        if "45" in txt: return 45.0
        if "180" in txt: return 180.0
        return 90.0

    def _update_phase_slider_ranges(self):
        r = self._phase_range_value()
        for (s, _, _, _) in self._phase_widgets:
            s.configure(from_=r, to=-r)

    def _add_phase_band(self):
        freqs = [b.f_hz for b in self.phase_bands] or [20.0, 20000.0]
        f_new = np.sqrt(min(freqs) * max(freqs))
        self.phase_bands.append(PhaseBand(f_new, 2.0, 0.0, False))
        self._rebuild_phase_sliders(); self._update_phase_slider_ranges(); self._maybe_preview()

    def _remove_last_phase_band(self):
        if self.phase_bands:
            self.phase_bands.pop()
            self._rebuild_phase_sliders(); self._update_phase_slider_ranges(); self._maybe_preview()

    def _reset_phase_gains(self):
        for b in self.phase_bands:
            b.gain_deg = 0.0
        for (s, _, _, _) in self._phase_widgets:
            s.set(0)

    def _on_phase_gain_changed(self, idx, val):
        try:
            self.phase_bands[idx].gain_deg = float(val)
        except Exception:
            pass
        self._maybe_preview()

    def _on_phase_freq_changed(self, idx, var):
        try:
            self.phase_bands[idx].f_hz = max(1.0, float(var.get()))
        except Exception:
            pass
        self._maybe_preview()

    def _on_phase_q_changed(self, idx, var):
        try:
            self.phase_bands[idx].Q = max(0.05, float(var.get()))
        except Exception:
            pass
        self._maybe_preview()

    def _on_phase_bypass(self, idx, by_var):
        self.phase_bands[idx].bypass = bool(by_var.get())
        self._maybe_preview()

    # --------- Preview / Export ---------
    def _pick_dir(self):
        d = filedialog.askdirectory(initialdir=self.dir_var.get() or os.path.expanduser("~"))
        if d:
            self.dir_var.set(d)

    def _get_preview_fft_len(self, auto_fft_len):
        if self.hi_res_preview_var.get():
            return auto_fft_len
        try:
            cap = int(self.preview_cap_var.get())
        except Exception:
            cap = 131072
        return min(auto_fft_len, cap)

    def _pow2ceil(self, n):
        n = int(max(1, n))
        return 1 << (n - 1).bit_length()

    def _snapshot_defs(self):
        """Collect current FIR UI state for persistence in the project."""
        def bands_to_list(bands, kind):
            out = []
            if kind == "mag":
                for b in bands:
                    out.append({"f_hz": float(b.f_hz), "Q": float(b.Q), "gain_db": float(b.gain_db), "bypass": bool(b.bypass)})
            else:
                for b in bands:
                    out.append({"f_hz": float(b.f_hz), "Q": float(b.Q), "gain_deg": float(b.gain_deg), "bypass": bool(b.bypass)})
            return out

        return {
            "fs": self.fs_var.get(),
            "taps": self.taps_var.get(),
            "fft": self.fft_var.get(),
            "auto_fft": bool(self.auto_fft_var.get()),
            "fft_mult": self.fft_mult_var.get(),
            "preview_cap": self.preview_cap_var.get(),
            "hi_res_preview": bool(self.hi_res_preview_var.get()),
            "centering": self.centering_var.get(),
            "custom_shift": self.custom_shift_var.get(),
            "window": self.window_var.get(),
            "filename": self.filename_var.get(),
            "directory": self.dir_var.get(),
            "live_preview": bool(self.live_preview.get()),
            "mag_range": self.mag_range_var.get(),
            "phase_range": self.phase_range_var.get(),
            "mag_bands": bands_to_list(self.mag_bands, "mag"),
            "phase_bands": bands_to_list(self.phase_bands, "phase"),
        }

    def _apply_snapshot_defs(self, snap):
        """Apply a previously saved snapshot to the UI and bands."""
        # Simple fields
        for key, var in [
            ("fs", self.fs_var),
            ("taps", self.taps_var),
            ("fft", self.fft_var),
            ("fft_mult", self.fft_mult_var),
            ("preview_cap", self.preview_cap_var),
            ("centering", self.centering_var),
            ("custom_shift", self.custom_shift_var),
            ("window", self.window_var),
            ("filename", self.filename_var),
            ("directory", self.dir_var),
            ("mag_range", self.mag_range_var),
            ("phase_range", self.phase_range_var),
        ]:
            if key in snap:
                var.set(str(snap[key]))

        for key, var in [
            ("auto_fft", self.auto_fft_var),
            ("hi_res_preview", self.hi_res_preview_var),
            ("live_preview", self.live_preview),
        ]:
            if key in snap:
                var.set(bool(snap[key]))

        # Bands
        def list_to_bands(lst, kind):
            out = []
            if not isinstance(lst, (list, tuple)):
                return out
            for d in lst:
                try:
                    f = float(d.get("f_hz", 1000.0))
                    Q = float(d.get("Q", 2.0))
                    if kind == "mag":
                        g = float(d.get("gain_db", 0.0)); bp = bool(d.get("bypass", False))
                        out.append(MagBand(f, Q, g, bp))
                    else:
                        g = float(d.get("gain_deg", 0.0)); bp = bool(d.get("bypass", False))
                        out.append(PhaseBand(f, Q, g, bp))
                except Exception:
                    continue
            return out

        if "mag_bands" in snap:
            self.mag_bands = list_to_bands(snap["mag_bands"], "mag") or self.mag_bands
        if "phase_bands" in snap:
            self.phase_bands = list_to_bands(snap["phase_bands"], "phase") or self.phase_bands

        # Rebuild sliders with new bands/ranges
        self._rebuild_mag_sliders(); self._rebuild_phase_sliders()
        self._update_mag_slider_ranges(); self._update_phase_slider_ranges()

    def _push_preview(self):
        try:
            fs = float(self.fs_var.get())
            # compute full auto-FFT length
            if self.auto_fft_var.get():
                taps = int(float(self.taps_var.get()))
                mult = {"x4": 4, "x8": 8, "x16": 16}.get(self.fft_mult_var.get(), 16)
                auto_len = self._pow2ceil(max(mult * max(1, taps), 8192))
                auto_len = min(auto_len, 1_048_576)
                nfft_preview = self._get_preview_fft_len(auto_len)
                self.fft_var.set(str(auto_len))
            else:
                auto_len = int(float(self.fft_var.get()))
                nfft_preview = self._get_preview_fft_len(auto_len)

            # build magnitude & phase targets (half spectrum), then combine
            _, Hph_half = build_phase_target(fs, int(nfft_preview), self.phase_bands)
            _, Amag_half = build_magnitude_target(fs, int(nfft_preview), self.mag_bands)  # real-positive
            H_half = Amag_half * Hph_half  # combine
            # update main for live preview
            self.master.set_fir_preview_for(self.which, fs, H_half)

            # let main capture our UI snapshot so Save Project embeds these controls
            try:
                self.master.set_fir_band_defs(self.which, self._snapshot_defs())
            except Exception:
                pass

            mode = "full" if self.hi_res_preview_var.get() else f"capped {int(nfft_preview)}"
            self.status.configure(text=f"Preview OK ({mode}). Export uses FFT={self.fft_var.get()}", foreground="#2a2")
        except Exception as e:
            self.status.configure(text=f"Preview error: {e}", foreground="#a22")

    def _maybe_preview(self, *_):
        if self.live_preview.get():
            self._push_preview()

    def _generate_csv(self):
        try:
            fs = float(self.fs_var.get())
            taps = int(float(self.taps_var.get()))
            if self.auto_fft_var.get():
                self._maybe_auto_fft()
            nfft = int(float(self.fft_var.get()))
            cent = self.centering_var.get()
            cshift = int(float(self.custom_shift_var.get())) if cent == "custom" else 0
            window = self.window_var.get()

            # full-res export (combined mag + phase)
            _, Hph_half = build_phase_target(fs, nfft, self.phase_bands)
            _, Amag_half = build_magnitude_target(fs, nfft, self.mag_bands)
            H_half = Amag_half * Hph_half

            h = synthesize_impulse_from_Hhalf(
                H_half, taps,
                window=window, centering=cent,
                custom_shift=cshift, normalize_dc=True
            )

            # CSV one-row with exactly N=taps values
            values = [f"{v:.14e}" for v in h.tolist()]
            csv_line = ",".join(values)
            fname = (self.filename_var.get() or f"impulse_{self.which.lower()}").strip()
            if not fname.lower().endswith(".csv"):
                fname += ".csv"
            out_path = os.path.join(self.dir_var.get() or ".", fname)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(csv_line + "\n")

            # after export, push a fresh preview and snapshot
            self.status.configure(text=f"Saved CSV: {out_path}", foreground="#2a2")
            self._push_preview()
        except Exception as e:
            self.status.configure(text=f"Error: {e}", foreground="#a22")

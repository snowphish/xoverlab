# gui_main.py

import os
import pickle
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from constants import (
    FREQ_MIN_DEFAULT,
    FREQ_MAX_DEFAULT,
)
from dsp import (
    parse_rew_txt,
    resample_complex_to_log_grid,
    apply_filter,
    smooth_fractional_octave_mag,
    smooth_fractional_octave_phase,
    wrap_phase_deg,
)

from fir_window import FIRWindow
from graph_view import GraphView


class XoverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XoverLab Lite")
        # keep everything visible without resizing
        self.geometry("1100x900")
        # prevent shrinking so far that legend overlaps plots
        self.minsize(1100, 900)

        # ---- Project path + FIR defs snapshot (from FIR window) ----
        self._project_path = None
        self._firA_defs = None   # FIR A band/settings snapshot (for saving/loading)
        self._firB_defs = None   # FIR B band/settings snapshot

        # ---- File menu for Save/Load (binary .xlab) ----
        menubar = tk.Menu(self)
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="Open…", accelerator="Ctrl+O", command=self.open_project)
        m_file.add_separator()
        m_file.add_command(label="Save", accelerator="Ctrl+S", command=self.save_project)
        m_file.add_command(label="Save As…", command=self.save_project_as)
        menubar.add_cascade(label="File", menu=m_file)
        self.config(menu=menubar)

        # Shortcuts
        self.bind_all("<Control-o>", lambda e: self.open_project())
        self.bind_all("<Control-s>", lambda e: self.save_project())
        self.bind_all("<Command-o>", lambda e: self.open_project())  # mac
        self.bind_all("<Command-s>", lambda e: self.save_project())

        # ---- State ----
        self.fg = None
        self.HA_raw = None
        self.HB_raw = None
        self.nameA = "(A)"
        self.nameB = "(B)"
        
        # Visibility toggles
        self.a_show = tk.BooleanVar(value=True)
        self.b_show = tk.BooleanVar(value=True)
        self.sum_show = tk.BooleanVar(value=True)

        # NEW: per-side raw trace toggles
        self.a_raw_show = tk.BooleanVar(value=True)
        self.b_raw_show = tk.BooleanVar(value=True)

        # default filters = none
        self.a_ftype = tk.StringVar(value="none")
        self.a_topo = tk.StringVar(value="LR24")
        self.a_fc = tk.StringVar(value="1000")
        self.b_ftype = tk.StringVar(value="none")
        self.b_topo = tk.StringVar(value="LR24")
        self.b_fc = tk.StringVar(value="1000")

        self.a_show = tk.BooleanVar(value=True)
        self.b_show = tk.BooleanVar(value=True)
        self.sum_show = tk.BooleanVar(value=True)
        

        self.a_gain = tk.StringVar(value="0")
        self.b_gain = tk.StringVar(value="0")
        self.a_delay = tk.StringVar(value="0")
        self.b_delay = tk.StringVar(value="0")
        self.a_invert = tk.BooleanVar(value=False)
        self.b_invert = tk.BooleanVar(value=False)

        self.smooth_choice = tk.StringVar(value="None")
        self.smooth_phase_too = tk.BooleanVar(value=True)

        self.freq_min_var = tk.StringVar(value=str(FREQ_MIN_DEFAULT))
        self.freq_max_var = tk.StringVar(value=str(FREQ_MAX_DEFAULT))
        self.mag_auto_var = tk.BooleanVar(value=True)
        self.mag_min_var = tk.StringVar(value="-60")
        self.mag_max_var = tk.StringVar(value="20")
        self._last_fmin = FREQ_MIN_DEFAULT
        self._last_fmax = FREQ_MAX_DEFAULT
        self._last_mag_manual = (-60.0, 20.0)

        # FIR previews (per-trace): store both mag & phase
        self.firA_enabled = tk.BooleanVar(value=False)
        self.firB_enabled = tk.BooleanVar(value=False)
        self._firA_freqs = None
        self._firA_mag_lin = None
        self._firA_phase_un = None
        self._firB_freqs = None
        self._firB_mag_lin = None
        self._firB_phase_un = None

        self._build_ui()

        # Graph view (Matplotlib) now lives in a separate class/file
        self.graph = GraphView(self.plot_frame)

    def _build_ui(self):
        # ===== Top bar (single row) =====
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # Left: Load
        load = ttk.Frame(top)
        load.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(load, text="Load Measurement A", command=self.load_A).grid(row=0, column=0, padx=4, pady=2, sticky="w")
        ttk.Button(load, text="Load Measurement B", command=self.load_B).grid(row=1, column=0, padx=4, pady=2, sticky="w")
        self.lblA = ttk.Label(load, text="A: (no file)")
        self.lblA.grid(row=0, column=1, sticky="w", padx=6)
        self.lblB = ttk.Label(load, text="B: (no file)")
        self.lblB.grid(row=1, column=1, sticky="w", padx=6)

        # Middle: FIR EQ (Enable toggles + Edit buttons)
        fir = ttk.LabelFrame(top, text="FIR EQ (per trace)")
        fir.pack(side=tk.LEFT, padx=10, pady=2)
        ttk.Checkbutton(fir, text="Enable A", variable=self.firA_enabled, command=self.update_plot).grid(row=0, column=0, padx=6, pady=2)
        ttk.Button(fir, text="Edit A…", width=10, command=lambda: self.open_fir_window('A')).grid(row=0, column=1, padx=4, pady=2)
        ttk.Checkbutton(fir, text="Enable B", variable=self.firB_enabled, command=self.update_plot).grid(row=1, column=0, padx=6, pady=2)
        ttk.Button(fir, text="Edit B…", width=10, command=lambda: self.open_fir_window('B')).grid(row=1, column=1, padx=4, pady=2)

        # Right: Smoothing & Sum
        sm = ttk.LabelFrame(top, text="Smoothing & Sum")
        sm.pack(side=tk.LEFT, padx=10, pady=2)
        ttk.Label(sm, text="Smoothing (1/N):").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        sm_opts = ["None", "1/1", "1/3", "1/6", "1/12", "1/24", "1/48"]
        sm_dd = ttk.Combobox(sm, values=sm_opts, width=6, state="readonly", textvariable=self.smooth_choice)
        sm_dd.grid(row=0, column=1, padx=4, pady=2)
        sm_dd.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        ttk.Checkbutton(sm, text="Smooth phase too", variable=self.smooth_phase_too, command=self.update_plot).grid(row=0, column=2, padx=8, pady=2)
        ttk.Checkbutton(sm, text="Show Sum", variable=self.sum_show, command=self.update_plot).grid(row=0, column=3, padx=8, pady=2)

        # ===== Crossovers row =====
        xover = ttk.Frame(self)
        xover.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 4))
        topo = ["None"] + [f"LR{d}" for d in (12, 24, 36, 48)] + [f"BW{d}" for d in (12, 24, 36, 48)]

        colA = ttk.LabelFrame(xover, text="Filters for A")
        colA.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(colA, text="Type:").grid(row=0, column=0, sticky="e", padx=2)
        ttk.Label(colA, text="Topology:").grid(row=0, column=2, sticky="e", padx=2)
        ttk.Label(colA, text="Fc (Hz):").grid(row=0, column=4, sticky="e", padx=2)
        a_type = ttk.Combobox(colA, values=["none", "lp", "hp"], state="readonly", textvariable=self.a_ftype, width=6)
        a_topo = ttk.Combobox(colA, values=topo, state="readonly", textvariable=self.a_topo, width=7)
        a_fc = ttk.Entry(colA, textvariable=self.a_fc, width=8)
        a_type.grid(row=0, column=1, padx=2, pady=2)
        a_topo.grid(row=0, column=3, padx=2, pady=2)
        a_fc.grid(row=0, column=5, padx=2, pady=2)
        for w in (a_type, a_topo):
            w.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        a_fc.bind("<Return>", lambda e: self.update_plot())
        a_fc.bind("<FocusOut>", lambda e: self.update_plot())

        colB = ttk.LabelFrame(xover, text="Filters for B")
        colB.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(colB, text="Type:").grid(row=0, column=0, sticky="e", padx=2)
        ttk.Label(colB, text="Topology:").grid(row=0, column=2, sticky="e", padx=2)
        ttk.Label(colB, text="Fc (Hz):").grid(row=0, column=4, sticky="e", padx=2)
        b_type = ttk.Combobox(colB, values=["none", "lp", "hp"], state="readonly", textvariable=self.b_ftype, width=6)
        b_topo = ttk.Combobox(colB, values=topo, state="readonly", textvariable=self.b_topo, width=7)
        b_fc = ttk.Entry(colB, textvariable=self.b_fc, width=8)
        b_type.grid(row=0, column=1, padx=2, pady=2)
        b_topo.grid(row=0, column=3, padx=2, pady=2)
        b_fc.grid(row=0, column=5, padx=2, pady=2)
        for w in (b_type, b_topo):
            w.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        b_fc.bind("<Return>", lambda e: self.update_plot())
        b_fc.bind("<FocusOut>", lambda e: self.update_plot())

        # NEW: “Show Raw” checkbox for A (sits on the next row)
        ttk.Checkbutton(colA, text="Show Raw", variable=self.a_raw_show, command=self.update_plot).grid(
            row=0, column=6, padx=6, pady=2, sticky="w"
        )       


        # NEW: “Show Raw” checkbox for B (next row)
        ttk.Checkbutton(colB, text="Show Raw", variable=self.b_raw_show, command=self.update_plot).grid(
            row=0, column=6, padx=6, pady=2, sticky="w"
        )



        # ===== Adjustments row =====
        adj = ttk.Frame(self)
        adj.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 4))

        adjA = ttk.LabelFrame(adj, text="Adjust A")
        adjA.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Checkbutton(adjA, text="Show A", variable=self.a_show, command=self.update_plot).grid(row=0, column=0, padx=4, pady=2)
        ttk.Checkbutton(adjA, text="Invert", variable=self.a_invert, command=self.update_plot).grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(adjA, text="Gain (dB):").grid(row=0, column=2, sticky="e")
        a_gain = ttk.Entry(adjA, textvariable=self.a_gain, width=7)
        a_gain.grid(row=0, column=3, padx=4)
        ttk.Label(adjA, text="Delay (ms):").grid(row=0, column=4, sticky="e")
        a_del = ttk.Entry(adjA, textvariable=self.a_delay, width=7)
        a_del.grid(row=0, column=5, padx=4)
        for w in (a_gain, a_del):
            w.bind("<Return>", lambda e: self.update_plot())
            w.bind("<FocusOut>", lambda e: self.update_plot())

        adjB = ttk.LabelFrame(adj, text="Adjust B")
        adjB.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Checkbutton(adjB, text="Show B", variable=self.b_show, command=self.update_plot).grid(row=0, column=0, padx=4, pady=2)
        ttk.Checkbutton(adjB, text="Invert", variable=self.b_invert, command=self.update_plot).grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(adjB, text="Gain (dB):").grid(row=0, column=2, sticky="e")
        b_gain = ttk.Entry(adjB, textvariable=self.b_gain, width=7)
        b_gain.grid(row=0, column=3, padx=4)
        ttk.Label(adjB, text="Delay (ms):").grid(row=0, column=4, sticky="e")
        b_del = ttk.Entry(adjB, textvariable=self.b_delay, width=7)
        b_del.grid(row=0, column=5, padx=4)
        for w in (b_gain, b_del):
            w.bind("<Return>", lambda e: self.update_plot())
            w.bind("<FocusOut>", lambda e: self.update_plot())

        # ===== Graph limits =====
        axis = ttk.LabelFrame(self, text="Graph Limits (Zoom)")
        axis.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 6))
        ttk.Label(axis, text="Fmin (Hz):").grid(row=0, column=0, sticky="e")
        fmin = ttk.Entry(axis, textvariable=self.freq_min_var, width=10)
        fmin.grid(row=0, column=1, padx=(4, 12))
        ttk.Label(axis, text="Fmax (Hz):").grid(row=0, column=2, sticky="e")
        fmax = ttk.Entry(axis, textvariable=self.freq_max_var, width=10)
        fmax.grid(row=0, column=3, padx=(4, 20))
        ttk.Checkbutton(axis, text="Auto magnitude Y", variable=self.mag_auto_var, command=self.update_plot).grid(row=0, column=4, padx=(0, 12))
        ttk.Label(axis, text="Mag min (dB):").grid(row=0, column=5, sticky="e")
        mmin = ttk.Entry(axis, textvariable=self.mag_min_var, width=8)
        mmin.grid(row=0, column=6, padx=(4, 8))
        ttk.Label(axis, text="Mag max (dB):").grid(row=0, column=7, sticky="e")
        mmax = ttk.Entry(axis, textvariable=self.mag_max_var, width=8)
        mmax.grid(row=0, column=8, padx=(4, 8))
        ttk.Button(axis, text="Apply", command=self.update_plot).grid(row=0, column=9, padx=(10, 6))
        ttk.Button(axis, text="Reset", command=self.reset_axes).grid(row=0, column=10, padx=(6, 0))
        for w in (fmin, fmax, mmin, mmax):
            w.bind("<Return>", lambda e: self.update_plot())
            w.bind("<FocusOut>", lambda e: self.update_plot())

        # ===== Plot area =====
        plot = ttk.Frame(self)
        plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.plot_frame = plot

    # ---- FIR per-trace: preview storage ----
    def clear_fir(self, which):
        if which == 'A':
            self._firA_freqs = None
            self._firA_mag_lin = None
            self._firA_phase_un = None
            self.firA_enabled.set(False)
        else:
            self._firB_freqs = None
            self._firB_mag_lin = None
            self._firB_phase_un = None
            self.firB_enabled.set(False)
        self.update_plot()

    def set_fir_preview_for(self, which, fs, H_half):
        kmax = len(H_half) - 1
        freqs = np.linspace(0.0, fs / 2.0, kmax + 1)
        phase_un = np.unwrap(np.angle(H_half))
        mag_lin = np.abs(H_half)
        if which == 'A':
            self._firA_freqs = freqs
            self._firA_phase_un = phase_un
            self._firA_mag_lin = mag_lin
        else:
            self._firB_freqs = freqs
            self._firB_phase_un = phase_un
            self._firB_mag_lin = mag_lin
        self.update_plot()

    def _fir_complex_at(self, which, freqs_query):
        if which == 'A':
            f = self._firA_freqs
            phi = self._firA_phase_un
            mag = self._firA_mag_lin
        else:
            f = self._firB_freqs
            phi = self._firB_phase_un
            mag = self._firB_mag_lin
        if f is None or phi is None or mag is None:
            return None
        fq = np.asarray(freqs_query)
        phi_i = np.interp(np.clip(fq, f[0], f[-1]), f, phi, left=phi[0], right=phi[-1])
        mag_i = np.interp(np.clip(fq, f[0], f[-1]), f, mag, left=mag[0], right=mag[-1])
        return mag_i * np.exp(1j * phi_i)

    def open_fir_window(self, which):
        FIRWindow(self, which)

    # Called by FIRWindow whenever its bands/settings change
    def set_fir_band_defs(self, which, defs_dict):
        if which == 'A':
            self._firA_defs = defs_dict
        else:
            self._firB_defs = defs_dict

    # ---- File I/O for measurements ----
    def load_A(self): self._load_generic(True)
    def load_B(self): self._load_generic(False)

    def _load_generic(self, is_A):
        path = filedialog.askopenfilename(
            title="Select REW TXT",
            filetypes=[("Text/FRD/CSV", "*.txt *.dat *.frd *.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            f, mdb, ph = parse_rew_txt(path)
            fg, H, _ = resample_complex_to_log_grid(
                f, mdb, ph, grid_n=2400, fmin=FREQ_MIN_DEFAULT, fmax=FREQ_MAX_DEFAULT
            )
            if self.fg is None:
                self.fg = fg
            elif not np.allclose(self.fg, fg):
                # align to existing grid
                x_src = np.log2(fg)
                x_dst = np.log2(self.fg)
                mag = np.abs(H)
                pha = np.angle(H)
                H_aligned = np.full_like(self.fg, np.nan + 1j * np.nan, dtype=np.complex128)
                finite = np.isfinite(mag) & np.isfinite(pha)
                if np.any(finite):
                    mag_i = np.interp(x_dst, x_src[finite], mag[finite], left=np.nan, right=np.nan)
                    pha_un = np.unwrap(pha[finite])
                    pha_i = np.interp(x_dst, x_src[finite], pha_un, left=np.nan, right=np.nan)
                    H_aligned = mag_i * np.exp(1j * pha_i)
                H = H_aligned
            if is_A:
                self.HA_raw = H
                self.nameA = os.path.basename(path)
                self.lblA.configure(text=f"A: {self.nameA}")
            else:
                self.HB_raw = H
                self.nameB = os.path.basename(path)
                self.lblB.configure(text=f"B: {self.nameB}")
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Import error", f"Failed to import {os.path.basename(path)}:\n{e}")

    # ---- helpers ----
    def _parse_float(self, var, fallback):
        try:
            return float(var.get())
        except:
            return fallback

    def _get_user_freq_limits(self):
        try:
            fmin = float(self.freq_min_var.get())
        except:
            fmin = self._last_fmin
        try:
            fmax = float(self.freq_max_var.get())
        except:
            fmax = self._last_fmax
        fmin = max(1.0, fmin)
        fmax = max(fmin * 1.001, fmax)
        self._last_fmin, self._last_fmax = fmin, fmax
        return fmin, fmax

    def _get_user_mag_limits(self, data_for_auto):
        if self.mag_auto_var.get():
            if (data_for_auto is None) or (len(data_for_auto) == 0) or (not np.any(np.isfinite(data_for_auto))):
                return None
            finite = data_for_auto[np.isfinite(data_for_auto)]
            dmin = float(np.nanmin(finite))
            dmax = float(np.nanmax(finite))
            pad = max(2.0, 0.05 * (dmax - dmin + 1e-6))
            return (dmin - pad, dmax + pad)
        try:
            ymin = float(self.mag_min_var.get())
        except:
            ymin = self._last_mag_manual[0]
        try:
            ymax = float(self.mag_max_var.get())
        except:
            ymax = self._last_mag_manual[1]
        if ymax <= ymin:
            ymax = ymin + 1.0
        self._last_mag_manual = (ymin, ymax)
        return (ymin, ymax)

    def reset_axes(self):
        self.freq_min_var.set(str(FREQ_MIN_DEFAULT))
        self.freq_max_var.set(str(FREQ_MAX_DEFAULT))
        self.mag_auto_var.set(True)
        self.mag_min_var.set("-60")
        self.mag_max_var.set("20")
        self._last_fmin = FREQ_MIN_DEFAULT
        self._last_fmax = FREQ_MAX_DEFAULT
        self._last_mag_manual = (-60.0, 20.0)
        self.update_plot()

    def _get_fc(self, var):
        try:
            fc = float(var.get())
            return fc if (FREQ_MIN_DEFAULT / 8 <= fc <= FREQ_MAX_DEFAULT * 4) else None
        except:
            return None

    def _apply_adjustments(self, H, freqs, gain_db, delay_ms, invert):
        if H is None:
            return None
        gain = 10.0 ** (gain_db / 20.0)
        delay_s = delay_ms / 1000.0
        out = np.full_like(H, np.nan + 1j * np.nan, dtype=np.complex128)
        ok = np.isfinite(H)
        if np.any(ok):
            phase_delay = np.exp(-1j * 2 * np.pi * freqs[ok] * delay_s)
            inv = -1.0 if invert else 1.0
            out[ok] = H[ok] * gain * phase_delay * inv
        return out

    def _to_magdb(self, H):
        if H is None:
            return None
        mag = np.abs(H)
        out = np.full_like(mag, np.nan, dtype=float)
        ok = np.isfinite(mag)
        out[ok] = 20 * np.log10(np.maximum(mag[ok], 1e-12))
        return out

    def _to_phdeg(self, H):
        if H is None:
            return None
        ang = np.angle(H)
        out = np.full_like(ang, np.nan, dtype=float)
        ok = np.isfinite(ang)
        out[ok] = wrap_phase_deg(np.rad2deg(ang[ok]))
        return out

    # ---- plot ----
    def update_plot(self):
        traces_loaded = (self.HA_raw is not None) or (self.HB_raw is not None)
        if self.fg is None or not traces_loaded:
            fmin, fmax = self._get_user_freq_limits()
            mag_ylim = self._get_user_mag_limits(np.array([-60, 20], dtype=float))
            # draw empty axes
            self.graph.render(
                freqs=np.array([fmin, fmax]),
                fmin=fmin, fmax=fmax,
                mag_ylim=mag_ylim,
                mag_items=[],
                phase_items=[],
            )
            return

        freqs = self.fg
        a_gain = self._parse_float(self.a_gain, 0.0)
        b_gain = self._parse_float(self.b_gain, 0.0)
        a_delay = self._parse_float(self.a_delay, 0.0)
        b_delay = self._parse_float(self.b_delay, 0.0)

        HA_adj = self._apply_adjustments(self.HA_raw, freqs, a_gain, a_delay, self.a_invert.get()) if self.HA_raw is not None else None
        HB_adj = self._apply_adjustments(self.HB_raw, freqs, b_gain, b_delay, self.b_invert.get()) if self.HB_raw is not None else None

        a_ftype, a_topo, a_fc = self.a_ftype.get().lower(), self.a_topo.get(), self._get_fc(self.a_fc)
        b_ftype, b_topo, b_fc = self.b_ftype.get().lower(), self.b_topo.get(), self._get_fc(self.b_fc)
        HA_f = apply_filter(HA_adj, freqs, a_ftype, a_topo, a_fc) if HA_adj is not None else None
        HB_f = apply_filter(HB_adj, freqs, b_ftype, b_topo, b_fc) if HB_adj is not None else None

        # Apply FIR per trace (mag + phase)
        if self.firA_enabled.get() and (self._firA_freqs is not None) and (HA_f is not None):
            HfirA = self._fir_complex_at('A', freqs)
            ok = np.isfinite(HA_f)
            HA_f = np.where(ok, HA_f * HfirA, HA_f)
        if self.firB_enabled.get() and (self._firB_freqs is not None) and (HB_f is not None):
            HfirB = self._fir_complex_at('B', freqs)
            ok = np.isfinite(HB_f)
            HB_f = np.where(ok, HB_f * HfirB, HB_f)

        # Sum after all processing
        if (HA_f is not None) and (HB_f is not None):
            Hsum = HA_f + HB_f
        elif HA_f is not None:
            Hsum = HA_f.copy()
        elif HB_f is not None:
            Hsum = HB_f.copy()
        else:
            Hsum = None

        # Convert to mag/phase
        magA_raw = self._to_magdb(HA_adj)
        magB_raw = self._to_magdb(HB_adj)
        magA_f   = self._to_magdb(HA_f)
        magB_f   = self._to_magdb(HB_f)
        magSum   = self._to_magdb(Hsum)
        phaA_raw = self._to_phdeg(HA_adj)
        phaB_raw = self._to_phdeg(HB_adj)
        phaA_f   = self._to_phdeg(HA_f)
        phaB_f   = self._to_phdeg(HB_f)
        phaSum   = self._to_phdeg(Hsum)

        # Smoothing
        frac = self.smooth_choice.get()
        if magA_raw is not None: magA_raw = smooth_fractional_octave_mag(freqs, magA_raw, frac)
        if magB_raw is not None: magB_raw = smooth_fractional_octave_mag(freqs, magB_raw, frac)
        if magA_f   is not None: magA_f   = smooth_fractional_octave_mag(freqs, magA_f, frac)
        if magB_f   is not None: magB_f   = smooth_fractional_octave_mag(freqs, magB_f, frac)
        if magSum   is not None: magSum   = smooth_fractional_octave_mag(freqs, magSum, frac)

        if self.smooth_phase_too.get():
            if phaA_raw is not None: phaA_raw = smooth_fractional_octave_phase(freqs, phaA_raw, frac)
            if phaB_raw is not None: phaB_raw = smooth_fractional_octave_phase(freqs, phaB_raw, frac)
            if phaA_f   is not None: phaA_f   = smooth_fractional_octave_phase(freqs, phaA_f, frac)
            if phaB_f   is not None: phaB_f   = smooth_fractional_octave_phase(freqs, phaB_f, frac)
            if phaSum   is not None: phaSum   = smooth_fractional_octave_phase(freqs, phaSum, frac)

        # Axis limits
        fmin, fmax = self._get_user_freq_limits()
        vis = (freqs >= fmin) & (freqs <= fmax)
        mags_for_auto = []
        for arr in (magA_raw, magB_raw, magA_f, magB_f, magSum):
            if arr is not None and np.any(vis):
                mags_for_auto.append(arr[vis])
        mag_for_auto = np.concatenate(mags_for_auto) if mags_for_auto else np.array([-60, 20], dtype=float)
        mag_ylim = self._get_user_mag_limits(mag_for_auto)

        # Shorter legend labels
        nameA = self.nameA or "A"
        nameB = self.nameB or "B"
        label_raw_A = f"raw {nameA}"
        label_flt_A = f"filtered {nameA}"
        label_raw_B = f"raw {nameB}"
        label_flt_B = f"filtered {nameB}"

        # Prepare draw lists for the GraphView
        mag_items = []
        phase_items = []

        if self.a_show.get() and self.a_raw_show.get() and magA_raw is not None:
            mag_items.append({"y": magA_raw, "alpha": 0.35, "lw": 1, "label": label_raw_A})
        if self.b_show.get() and self.b_raw_show.get() and magB_raw is not None:
            mag_items.append({"y": magB_raw, "alpha": 0.35, "lw": 1, "label": label_raw_B})

        if self.a_show.get() and magA_f is not None:
            mag_items.append({"y": magA_f, "lw": 1.6, "label": label_flt_A})
        if self.b_show.get() and magB_f is not None:
            mag_items.append({"y": magB_f, "lw": 1.6, "label": label_flt_B})
        if self.sum_show.get() and magSum is not None:
            mag_items.append({"y": magSum, "lw": 2.0, "label": "Sum"})

        if self.a_show.get() and phaA_raw is not None:
            phase_items.append({"y": phaA_raw, "alpha": 0.35, "lw": 1, "label": label_raw_A})
        if self.b_show.get() and phaB_raw is not None:
            phase_items.append({"y": phaB_raw, "alpha": 0.35, "lw": 1, "label": label_raw_B})
        if self.a_show.get() and phaA_f is not None:
            phase_items.append({"y": phaA_f, "lw": 1.6, "label": label_flt_A})
        if self.b_show.get() and phaB_f is not None:
            phase_items.append({"y": phaB_f, "lw": 1.6, "label": label_flt_B})
        if self.sum_show.get() and phaSum is not None:
            phase_items.append({"y": phaSum, "lw": 2.0, "label": "Sum"})

        # Render via GraphView
        self.graph.render(
            freqs=freqs,
            fmin=fmin, fmax=fmax,
            mag_ylim=mag_ylim,
            mag_items=mag_items,
            phase_items=phase_items,
        )

    # ========= Binary Project Save/Load (.xlab with pickle) =========

    def _collect_state(self):
        """Gather everything needed to fully restore the session."""
        ui_state = {
            "version": 1,
            "saved_at_utc": datetime.datetime.utcnow().isoformat() + "Z",

            # Names / labels
            "nameA": self.nameA,
            "nameB": self.nameB,

            # Filters
            "a_ftype": self.a_ftype.get(),
            "a_topo": self.a_topo.get(),
            "a_fc": self.a_fc.get(),
            "b_ftype": self.b_ftype.get(),
            "b_topo": self.b_topo.get(),
            "b_fc": self.b_fc.get(),

            # Visibility / sum
            "a_show": bool(self.a_show.get()),
            "b_show": bool(self.b_show.get()),
            "sum_show": bool(self.sum_show.get()),

            # Adjustments
            "a_gain": self.a_gain.get(),
            "a_delay": self.a_delay.get(),
            "a_invert": bool(self.a_invert.get()),
            "b_gain": self.b_gain.get(),
            "b_delay": self.b_delay.get(),
            "b_invert": bool(self.b_invert.get()),

            # Smoothing / axes
            "smooth_choice": self.smooth_choice.get(),
            "smooth_phase_too": bool(self.smooth_phase_too.get()),
            "freq_min": self.freq_min_var.get(),
            "freq_max": self.freq_max_var.get(),
            "mag_auto": bool(self.mag_auto_var.get()),
            "mag_min": self.mag_min_var.get(),
            "mag_max": self.mag_max_var.get(),

            # FIR toggles
            "firA_enabled": bool(self.firA_enabled.get()),
            "firB_enabled": bool(self.firB_enabled.get()),
        }

        payload = {
            "ui": ui_state,
            "fg": self.fg,
            "HA_raw": self.HA_raw,
            "HB_raw": self.HB_raw,
            "firA_preview": {"freqs": self._firA_freqs, "mag": self._firA_mag_lin, "phase": self._firA_phase_un},
            "firB_preview": {"freqs": self._firB_freqs, "mag": self._firB_mag_lin, "phase": self._firB_phase_un},
            "firA_defs": self._firA_defs,
            "firB_defs": self._firB_defs,
        }
        return payload

    def _apply_state(self, payload):
        """Restore UI + arrays from a saved payload."""
        ui = payload.get("ui", {})

        # Names
        self.nameA = ui.get("nameA", self.nameA)
        self.nameB = ui.get("nameB", self.nameB)
        if hasattr(self, "lblA"):
            self.lblA.configure(text=f"A: {self.nameA}")
        if hasattr(self, "lblB"):
            self.lblB.configure(text=f"B: {self.nameB}")

        # Filters
        self.a_ftype.set(ui.get("a_ftype", self.a_ftype.get()))
        self.a_topo.set(ui.get("a_topo", self.a_topo.get()))
        self.a_fc.set(ui.get("a_fc", self.a_fc.get()))
        self.b_ftype.set(ui.get("b_ftype", self.b_ftype.get()))
        self.b_topo.set(ui.get("b_topo", self.b_topo.get()))
        self.b_fc.set(ui.get("b_fc", self.b_fc.get()))

        # Visibility / sum
        self.a_show.set(ui.get("a_show", True))
        self.b_show.set(ui.get("b_show", True))
        self.sum_show.set(ui.get("sum_show", True))

        # Adjustments
        self.a_gain.set(ui.get("a_gain", "0"))
        self.a_delay.set(ui.get("a_delay", "0"))
        self.a_invert.set(ui.get("a_invert", False))
        self.b_gain.set(ui.get("b_gain", "0"))
        self.b_delay.set(ui.get("b_delay", "0"))
        self.b_invert.set(ui.get("b_invert", False))

        # Smoothing / axes
        self.smooth_choice.set(ui.get("smooth_choice", "None"))
        self.smooth_phase_too.set(ui.get("smooth_phase_too", True))
        self.freq_min_var.set(ui.get("freq_min", self.freq_min_var.get()))
        self.freq_max_var.set(ui.get("freq_max", self.freq_max_var.get()))
        self.mag_auto_var.set(ui.get("mag_auto", True))
        self.mag_min_var.set(ui.get("mag_min", "-60"))
        self.mag_max_var.set(ui.get("mag_max", "20"))

        # FIR toggles
        self.firA_enabled.set(ui.get("firA_enabled", False))
        self.firB_enabled.set(ui.get("firB_enabled", False))

        # Arrays
        self.fg = payload.get("fg", None)
        self.HA_raw = payload.get("HA_raw", None)
        self.HB_raw = payload.get("HB_raw", None)

        # Previews
        firA_prev = payload.get("firA_preview", {}) or {}
        firB_prev = payload.get("firB_preview", {}) or {}
        self._firA_freqs = firA_prev.get("freqs", None)
        self._firA_mag_lin = firA_prev.get("mag", None)
        self._firA_phase_un = firA_prev.get("phase", None)
        self._firB_freqs = firB_prev.get("freqs", None)
        self._firB_mag_lin = firB_prev.get("mag", None)
        self._firB_phase_un = firB_prev.get("phase", None)

        # FIR editor inputs snapshot
        self._firA_defs = payload.get("firA_defs", None)
        self._firB_defs = payload.get("firB_defs", None)

        # Redraw
        self.update_plot()

    def save_project(self):
        if not self._project_path:
            return self.save_project_as()
        state = self._collect_state()
        try:
            with open(self._project_path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save project:\n{e}")

    def save_project_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".xlab",
            filetypes=[("XoverLab Project", "*.xlab"), ("All files", "*.*")],
        )
        if not path:
            return
        self._project_path = path
        self.save_project()
        messagebox.showinfo("Saved", f"Project saved:\n{path}")

    def open_project(self):
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("XoverLab Project", "*.xlab"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)  # local trusted file
            self._project_path = path
            self._apply_state(payload)
            messagebox.showinfo("Opened", f"Project loaded:\n{path}")
        except Exception as e:
            messagebox.showerror("Open error", f"Failed to open project:\n{e}")

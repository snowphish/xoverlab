# graph_view.py
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from constants import FREQ_TICKS, fmt_freq_tick


class GraphView:
    """
    Two stacked axes (Magnitude & Phase) with:
      • a single shared legend in a padded band between them
      • synced vertical crosshair across both plots
      • live readout showing ONLY the current frequency

    Legend gap resizes automatically (with generous padding),
    and top margin ensures the Magnitude title never clips.
    """

    def __init__(self, parent):
        # Headroom so the top title never clips
        self.fig = Figure(figsize=(11.8, 7.6), dpi=100)
        self.fig.subplots_adjust(right=0.97, top=0.94, bottom=0.085, left=0.07, hspace=0.12)

        self.ax_mag = self.fig.add_subplot(2, 1, 1)
        self.ax_phase = self.fig.add_subplot(2, 1, 2)

        # Axis used only to host the legend
        self.leg_ax = self.fig.add_axes([0, 0, 0, 0], frameon=False)
        self.leg_ax.axis('off')
        self._legend = None

        # Tk canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Baseline positions so every render starts cleanly
        self._base_pos_top = self.ax_mag.get_position().frozen()
        self._base_pos_bot = self.ax_phase.get_position().frozen()

        # Layout tuning (figure coords)
        self._min_gap = 0.06          # minimum total gap between axes
        self._legend_pad = 0.018      # padding inside legend band
        self._legend_air_top = 0.020  # space between legend and Magnitude
        self._legend_air_bot = 0.030  # space between legend and Phase

        # ==== Cursor + readout state ====
        self.vline_mag = None
        self.vline_phase = None
        self.readout_mag = None
        self.readout_phase = None

        # Connect events
        self.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.canvas.mpl_connect('figure_leave_event', self._on_leave)
        self.canvas.mpl_connect('axes_leave_event', self._on_axes_leave)

    # ---------- axes setup ----------
    def _setup_axes(self, fmin, fmax, mag_ylim):
        self.ax_mag.set_title("Magnitude (dB)")
        self.ax_mag.set_xscale("log")
        self.ax_mag.set_xlim(max(1.0, fmin), max(1.1, fmax))
        if mag_ylim is not None:
            self.ax_mag.set_ylim(mag_ylim[0], mag_ylim[1])
        self.ax_mag.grid(True, which="both", ls=":", alpha=0.5)
        xt = [t for t in FREQ_TICKS if fmin <= t <= fmax] or [fmin, fmax]
        self.ax_mag.set_xticks(xt)
        self.ax_mag.set_xticklabels([fmt_freq_tick(x) for x in xt])

        self.ax_phase.set_title("Phase (deg)  [wrapped -180° .. +180°]")
        self.ax_phase.set_xscale("log")
        self.ax_phase.set_xlim(max(1.0, fmin), max(1.1, fmax))
        self.ax_phase.set_ylim(-180, 180)
        self.ax_phase.grid(True, which="both", ls=":", alpha=0.5)
        self.ax_phase.set_xticks(xt)
        self.ax_phase.set_xticklabels([fmt_freq_tick(x) for x in xt])

    # ---------- legend helpers ----------
    def _max_bottom_xtick_height_fig(self, ax, renderer):
        hs = []
        for lab in ax.get_xticklabels():
            if not lab.get_visible():
                continue
            bb = lab.get_window_extent(renderer=renderer)
            hs.append(bb.transformed(self.fig.transFigure.inverted()).height)
        return max(hs) if hs else 0.0

    def _title_height_fig(self, ax, renderer):
        t = ax.title
        if not t or not t.get_text():
            return 0.0
        bb = t.get_window_extent(renderer=renderer)
        return bb.transformed(self.fig.transFigure.inverted()).height

    def _build_shared_legend(self):
        handles, labels = self.ax_mag.get_legend_handles_labels()

        # Deduplicate by label
        uniq, seen = [], set()
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append((h, l))
                seen.add(l)
        handles = [h for h, _ in uniq]
        labels = [l for _, l in uniq]

        # First-pass placement; final position after measuring
        pos_top = self.ax_mag.get_position()
        pos_bot = self.ax_phase.get_position()
        x0, x1 = pos_top.x0, pos_top.x1
        y0, y1 = pos_bot.y1, pos_top.y0
        gap_h = max(self._min_gap, (y1 - y0) * 0.9)
        y = y0 + ((y1 - y0) - gap_h) / 2.0

        self.leg_ax.set_position([x0, y, x1 - x0, gap_h])
        self.leg_ax.cla()
        self.leg_ax.axis('off')

        self._legend = None
        if handles:
            # Single row to save vertical space
            self._legend = self.leg_ax.legend(
                handles, labels,
                loc="center",
                ncol=len(labels),
                frameon=True, fancybox=True, framealpha=0.95,
                borderpad=0.6, labelspacing=0.7, columnspacing=1.4, handlelength=2.6,
                prop={"size": "small"},
            )

    def _layout_to_fit_legend(self):
        if self._legend is None:
            return

        renderer = self.canvas.get_renderer() or (self.canvas.draw() or self.canvas.get_renderer())

        # Height budget (figure coords)
        leg_h = self._legend.get_window_extent(renderer=renderer).transformed(
            self.fig.transFigure.inverted()).height
        xtick_h = self._max_bottom_xtick_height_fig(self.ax_mag, renderer)  # below Magnitude
        title_h = self._title_height_fig(self.ax_phase, renderer)           # above Phase

        leg_band_h = leg_h + 2 * self._legend_pad
        needed_gap = max(
            self._min_gap,
            leg_band_h + xtick_h + title_h + self._legend_air_top + self._legend_air_bot
        )

        # Current positions
        pos_top = self.ax_mag.get_position().frozen()
        pos_bot = self.ax_phase.get_position().frozen()
        current_gap = pos_top.y0 - pos_bot.y1

        def place_legend(gap_bottom_y, gap_top_y):
            """Position legend inside [gap_bottom_y, gap_top_y] with min air kept; center any extra."""
            gap = gap_top_y - gap_bottom_y
            extra = max(0.0, gap - (leg_band_h + self._legend_air_top + self._legend_air_bot + xtick_h + title_h))
            leg_y = gap_bottom_y + self._legend_air_bot + (title_h) + extra / 2.0
            self.leg_ax.set_position([pos_top.x0, leg_y, pos_top.x1 - pos_top.x0, leg_band_h])

        if current_gap >= needed_gap:
            place_legend(pos_bot.y1, pos_top.y0)
            return

        # Not enough space: shrink axes to open the required gap
        total_room = (pos_top.y1 - pos_bot.y0)
        needed_gap = min(needed_gap, max(0.30, total_room * 0.40))
        remaining = max(0.08, total_room - needed_gap)

        h_top, h_bot = pos_top.height, pos_bot.height
        denom = max(1e-9, (h_top + h_bot))
        new_h_top = remaining * (h_top / denom)
        new_h_bot = remaining * (h_bot / denom)

        # Anchor extremes
        new_top_y1 = pos_top.y1
        new_top_y0 = new_top_y1 - new_h_top
        new_bot_y0 = pos_bot.y0
        new_bot_y1 = new_bot_y0 + new_h_bot

        # Apply and place legend
        self.ax_mag.set_position([pos_top.x0, new_top_y0, pos_top.width, new_h_top])
        self.ax_phase.set_position([pos_bot.x0, new_bot_y0, pos_bot.width, new_h_bot])
        place_legend(new_bot_y1, new_top_y0)

    # ---------- cursor helpers ----------
    def _ensure_crosshair(self):
        # Create fresh crosshairs if missing
        if self.vline_mag is None:
            (self.vline_mag,) = self.ax_mag.plot(
                [1, 1], [0, 1],
                linestyle="--", linewidth=0.8, alpha=0.7,
                transform=self.ax_mag.get_xaxis_transform(),
                zorder=10,
            )
        if self.vline_phase is None:
            (self.vline_phase,) = self.ax_phase.plot(
                [1, 1], [0, 1],
                linestyle="--", linewidth=0.8, alpha=0.7,
                transform=self.ax_phase.get_xaxis_transform(),
                zorder=10,
            )

    def _ensure_readouts(self):
        if self.readout_mag is None:
            self.readout_mag = self.ax_mag.text(
                0.995, 0.98, "", transform=self.ax_mag.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9),
                fontsize="small", zorder=11,
            )
        if self.readout_phase is None:
            self.readout_phase = self.ax_phase.text(
                0.995, 0.98, "", transform=self.ax_phase.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9),
                fontsize="small", zorder=11,
            )

    @staticmethod
    def _fmt_freq_value(x):
        if x is None or not np.isfinite(x):
            return ""
        if x < 1000:
            return f"{x:.0f} Hz"
        if x < 10000:
            return f"{x/1000.0:.2f} kHz"
        return f"{x/1000.0:.1f} kHz"

    # ---------- event handlers ----------
    def _on_move(self, event):
        if event.inaxes not in (self.ax_mag, self.ax_phase):
            return
        x = event.xdata
        if x is None or not np.isfinite(x):
            return

        self._ensure_crosshair()
        self._ensure_readouts()

        # Move crosshair on both axes
        self.vline_mag.set_xdata([x, x])
        self.vline_phase.set_xdata([x, x])

        # Update readout only in the active axes – show frequency only
        freq_txt = f"f: {self._fmt_freq_value(x)}"
        if event.inaxes is self.ax_mag:
            self.readout_mag.set_text(freq_txt)
            self.readout_mag.set_visible(True)
            if self.readout_phase:
                self.readout_phase.set_visible(False)
        else:
            self.readout_phase.set_text(freq_txt)
            self.readout_phase.set_visible(True)
            if self.readout_mag:
                self.readout_mag.set_visible(False)

        self.canvas.draw_idle()

    def _on_axes_leave(self, _event):
        if self.readout_mag:
            self.readout_mag.set_visible(False)
        if self.readout_phase:
            self.readout_phase.set_visible(False)
        self.canvas.draw_idle()

    def _on_leave(self, _event):
        # Hide everything when leaving the figure
        if self.vline_mag:
            self.vline_mag.set_visible(False)
        if self.vline_phase:
            self.vline_phase.set_visible(False)
        if self.readout_mag:
            self.readout_mag.set_visible(False)
        if self.readout_phase:
            self.readout_phase.set_visible(False)
        self.canvas.draw_idle()

    # ---------- public API ----------
    def render(self, freqs, fmin, fmax, mag_ylim, mag_items, phase_items):
        """Clear, set axes, draw items, update legend, adjust layout, and redraw."""
        # Reset to baseline so layout is stable
        self.ax_mag.set_position(self._base_pos_top)
        self.ax_phase.set_position(self._base_pos_bot)

        # Clear axes (this deletes any existing artists on them)
        self.ax_mag.cla()
        self.ax_phase.cla()

        # IMPORTANT: since cla() removed old artists, drop our references
        self.vline_mag = None
        self.vline_phase = None
        self.readout_mag = None
        self.readout_phase = None

        self._setup_axes(fmin, fmax, mag_ylim)

        # Draw magnitude traces
        for item in mag_items:
            y = item.get("y")
            if y is None:
                continue
            kw = {k: v for k, v in item.items() if k != "y"}
            self.ax_mag.plot(freqs, y, **kw)

        # Draw phase traces
        for item in phase_items:
            y = item.get("y")
            if y is None:
                continue
            kw = {k: v for k, v in item.items() if k != "y"}
            self.ax_phase.plot(freqs, y, **kw)

        # Final limits
        self.ax_mag.set_xlim(fmin, fmax)
        self.ax_phase.set_xlim(fmin, fmax)
        if mag_ylim is not None:
            self.ax_mag.set_ylim(*mag_ylim)

        # Build legend, draw once to measure, then expand gap and redraw
        self._build_shared_legend()
        self.canvas.draw()
        self._layout_to_fit_legend()
        self.canvas.draw()

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve, Box2DKernel
from scipy import interpolate


__version__ = "0.1.0"


class ResolvitPaths:
    def __init__(self, events_list):

        self.events_list = Path(events_list)

        self.channel_dir = self.events_list.parent

        # uvit/
        self.uvit_dir = self.events_list.parents[2]

        self.resolvit_products_dir = self.uvit_dir / "resolvit_data_products"

        self.channel_name = self.channel_dir.name

        self.resolvit_channel_dir = self.resolvit_products_dir / self.channel_name

        self.resolvit_channel_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.product_id = self.events_list.name.replace("_l2ce.fits", "")

        self.diagnostics_dir = (
            self.resolvit_products_dir / "diagnostics" / self.product_id
        )

        self.diagnostics_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.corrected_events_list = self.resolvit_channel_dir / self.events_list.name

    def correlation_plot(
        self,
        iteration_no,
        bin_size,
        t_mid,
    ):
        return (
            self.diagnostics_dir
            / f"{iteration_no}_{bin_size:.0f}_{t_mid:.0f}_correlations.png"
        )

    def residual_plot(
        self,
        iteration_no,
    ):
        return self.diagnostics_dir / f"residuals_iteration_{iteration_no}.png"

    def residual_file(
        self,
        iteration_no,
    ):
        return self.diagnostics_dir / f"residuals_iteration_{iteration_no}.txt"


def read_columns(events_list_hdu):
    time = events_list_hdu[1].data["MJD_L2"]
    fx = events_list_hdu[1].data["Fx"]
    fy = events_list_hdu[1].data["Fy"]
    photons = events_list_hdu[1].data["EFFECTIVE_NUM_PHOTONS"]
    bad_flag = events_list_hdu[1].data["BAD FLAG"]
    mask = photons > 0
    mask = np.logical_and(mask, bad_flag)
    time = time[mask]
    fx = fx[mask]
    fy = fy[mask]
    photons = photons[mask]
    photons = photons / np.median(photons)
    return time, fx, fy, photons


class EventListCorrelator:
    def __init__(self, reference_events_list, to_match_events_list):
        self.reference_events_list = reference_events_list
        self.to_match_events_list = to_match_events_list
        self.upper_limit = 10

    def get_columns(self, data):
        time = data["time"]
        fx = data["fx"]
        fy = data["fy"]
        photons = data["photons"]
        return time, fx, fy, photons

    def get_image(self, hdu, bins):
        time, fx, fy, photons = self.get_columns(hdu)
        ndarray, yedges, xedges = np.histogram2d(
            fy, fx, bins=(bins, bins), weights=photons
        )
        return ndarray

    def get_lag(self, correlations, shift_range):
        f = interpolate.interp1d(shift_range, correlations, kind="quadratic")
        new_range = np.linspace(
            shift_range[0],
            shift_range[-1],
            num=(self.upper_limit * 2 + 1) * 100,
            endpoint=True,
        )
        interpolated_correlations = f(new_range)
        max_index = np.argmax(interpolated_correlations)
        peak_position = new_range[max_index]
        return peak_position

    def get_shifts(self, bin_size=1, bg_bin_size=64, correlation_plot=None):
        # To get shifts
        bins = np.arange(0, 4801, bin_size)
        reference_image = self.get_image(self.reference_events_list, bins)
        to_match_image = self.get_image(self.to_match_events_list, bins)

        box_kernel = Box2DKernel(5)
        boxed_reference_image = convolve(reference_image, box_kernel)
        reference_image_background_mask = boxed_reference_image <= (4 / 25)
        reference_image[reference_image_background_mask] = 0

        boxed_to_match_image = convolve(to_match_image, box_kernel)
        to_match_image_background_mask = boxed_to_match_image <= (4 / 25)
        to_match_image[to_match_image_background_mask] = 0

        kernel = Gaussian2DKernel(x_stddev=1)
        smoothed_reference_image = convolve(reference_image, kernel)
        smoothed_to_match_image = convolve(to_match_image, kernel)

        smoothed_reference_image = smoothed_reference_image / np.std(
            smoothed_reference_image
        )
        smoothed_to_match_image = smoothed_to_match_image / np.std(
            smoothed_to_match_image
        )

        smoothed_reference_image[smoothed_reference_image < 0] = 0
        smoothed_to_match_image[smoothed_to_match_image < 0] = 0

        smoothed_reference_X = np.sum(smoothed_reference_image, axis=0)
        smoothed_reference_Y = np.sum(smoothed_reference_image, axis=1)

        smoothed_to_match_X = np.sum(smoothed_to_match_image, axis=0)
        smoothed_to_match_Y = np.sum(smoothed_to_match_image, axis=1)

        shift_limit = int(self.upper_limit / bin_size)
        shift_range = np.arange(-shift_limit, shift_limit + 1)

        X_correlations = []
        Y_correlations = []
        for shift in shift_range:
            X_product = np.sum(
                smoothed_reference_X * np.roll(smoothed_to_match_X, shift)
            )
            Y_product = np.sum(
                smoothed_reference_Y * np.roll(smoothed_to_match_Y, shift)
            )
            X_correlations.append(X_product)
            Y_correlations.append(Y_product)

        x_shift = self.get_lag(X_correlations, shift_range) * bin_size
        y_shift = self.get_lag(Y_correlations, shift_range) * bin_size

        fig, ax = plt.subplots()
        ax.plot(shift_range, X_correlations, label="X correlations")
        ax.plot(shift_range, Y_correlations, label="Y correlations")
        ax.set_xlabel("Shifts in sub-pixels")
        ax.set_ylabel("Correlation")
        ax.legend()
        if correlation_plot is not None:
            fig.savefig(correlation_plot, dpi=150)
        plt.close(fig)

        return x_shift, y_shift


def plot_residuals(residuals, figure_name):
    t_array = np.array([(b["t_start"] + b["t_end"]) / 2 for b in residuals])
    dx_array = np.array([b["dx"] for b in residuals])
    dy_array = np.array([b["dy"] for b in residuals])

    fig, ax = plt.subplots()
    ax.plot(t_array, dx_array, label="X", marker=".", alpha=0.8)
    ax.plot(t_array, dy_array, label="Y", marker=".", alpha=0.8)

    ax.set_ylim(-3, 3)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residual (sub-pixels)")
    ax.legend()

    fig.savefig(figure_name, dpi=150)
    plt.close(fig)


def save_residuals_to_file(residuals, filename):
    data = np.array(
        [
            [r["t_start"], r["t_end"], r["total_events"], r["dx"], r["dy"]]
            for r in residuals
        ]
    )

    np.savetxt(
        filename,
        data,
        header="t_start t_end total_events dx dy",
        fmt=["%.6f", "%.6f", "%d", "%.6f", "%.6f"],
    )


def calculate_residuals(
    events_list,
    bin_size,
    start_offset,
    total_events_fraction,
    iteration_no,
    paths,
):
    events_list_hdu = fits.open(events_list)
    time, fx, fy, photons = read_columns(events_list_hdu)

    # Define bin edges
    t_start = time.min() + start_offset
    t_end = time.max()
    bins = np.arange(t_start, t_end + bin_size, bin_size)

    # Digitize: assign each event to a bin index
    bin_indices = np.digitize(time, bins)

    binned_data = []
    for i in range(1, len(bins)):
        mask = bin_indices == i

        if np.any(mask):
            binned_data.append(
                {
                    "t_start": bins[i - 1],
                    "t_end": bins[i],
                    "total_events": len(fx[mask]),
                    "time": time[mask],
                    "fx": fx[mask],
                    "fy": fy[mask],
                    "photons": photons[mask],
                }
            )

    total_events = np.array([b["total_events"] for b in binned_data])
    total_events_threshold = np.median(total_events) * total_events_fraction

    reference_data = None
    residuals = []
    for data in binned_data:
        if data["total_events"] > total_events_threshold:
            if reference_data is None:
                reference_data = {
                    "time": data["time"],
                    "fx": data["fx"],
                    "fy": data["fy"],
                    "photons": data["photons"],
                }

                residuals.append(
                    {
                        "t_start": data["t_start"],
                        "t_end": data["t_end"],
                        "total_events": data["total_events"],
                        "dx": 0,
                        "dy": 0,
                    }
                )
            else:
                to_match_data = {
                    "time": data["time"],
                    "fx": data["fx"],
                    "fy": data["fy"],
                    "photons": data["photons"],
                }

                init_correlation = EventListCorrelator(reference_data, to_match_data)
                t_mid = (data["t_start"] + data["t_end"]) / 2
                dx, dy = init_correlation.get_shifts(
                    correlation_plot=paths.correlation_plot(
                        iteration_no,
                        bin_size,
                        t_mid,
                    )
                )

                residuals.append(
                    {
                        "t_start": data["t_start"],
                        "t_end": data["t_end"],
                        "total_events": data["total_events"],
                        "dx": dx,
                        "dy": dy,
                    }
                )

    plot_residuals(
        residuals,
        paths.residual_plot(iteration_no),
    )

    save_residuals_to_file(
        residuals,
        paths.residual_file(iteration_no),
    )

    return residuals


def apply_residual_corrections(events_list, residuals, corrected_events_list, bin_size):
    events_list_hdu = fits.open(events_list)
    time = events_list_hdu[1].data["MJD_L2"]
    fx = events_list_hdu[1].data["Fx"]
    fy = events_list_hdu[1].data["Fy"]
    dx_events = np.zeros_like(fx)
    dy_events = np.zeros_like(fx)

    dx_array = np.array([b["dx"] for b in residuals])
    dy_array = np.array([b["dy"] for b in residuals])
    total_events_array = np.array([b["total_events"] for b in residuals])

    offset_dx = np.average(dx_array, weights=total_events_array)
    offset_dy = np.average(dy_array, weights=total_events_array)

    for r in residuals:
        t0 = r["t_start"]
        t1 = r["t_end"]
        dx = r["dx"] - offset_dx
        dy = r["dy"] - offset_dy

        mask = (time >= t0) & (time < t1)

        dx_events[mask] = dx
        dy_events[mask] = dy

    fx_corr = fx + dx_events
    fy_corr = fy + dy_events

    events_list_hdu[1].data["Fx"] = fx_corr
    events_list_hdu[1].data["Fy"] = fy_corr

    events_list_hdu[0].header["RESOLVIT"] = True
    events_list_hdu[0].header["RSLV_VER"] = __version__
    events_list_hdu[0].header["RSLV_BIN"] = bin_size

    events_list_hdu.writeto(corrected_events_list, overwrite=True)


def process_observation(
    observation_dir,
    bin_size=100.0,
    iteration_offsets=None,
    total_events_fraction=0.75,
):

    if iteration_offsets is None:
        iteration_offsets = [0, 1 / 2, 1 / 4, 1 / 3]

    events_lists = glob(
        f"{observation_dir}/**/data_products/*_data/**/AS1*l2ce.fits",
        recursive=True,
    )

    events_lists = sorted(events_lists)

    for events_list in events_lists:
        process_events_list(
            events_list, bin_size, iteration_offsets, total_events_fraction
        )


def process_events_list(
    events_list, bin_size, iteration_offsets, total_events_fraction
):

    print(f"Processing {Path(events_list).name}")

    paths = ResolvitPaths(events_list)

    current_file = events_list

    for i, frac in enumerate(
        iteration_offsets,
        start=1,
    ):

        offset = bin_size * frac

        residuals = calculate_residuals(
            current_file,
            bin_size,
            offset,
            total_events_fraction,
            i,
            paths,
        )

        apply_residual_corrections(
            current_file, residuals, paths.corrected_events_list, bin_size
        )

        current_file = paths.corrected_events_list

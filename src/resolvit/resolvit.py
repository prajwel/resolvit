#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from shutil import copy
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve, Box2DKernel
from datetime import datetime
from scipy import interpolate


__version__ = "0.2.2"

DEFAULT_BIN_SIZE = 100.0
DEFAULT_ITERATION_OFFSETS = [0, 1 / 2, 1 / 4, 1 / 3]
DEFAULT_TOTAL_EVENTS_FRACTION = 0.75
DEFAULT_UPPER_LIMIT = 10  # sub-pixels


def write_processing_log(
    paths,
    events_list,
    bin_size,
    total_events_fraction,
    iteration_offsets,
):
    with open(paths.log_file, "w") as logfile:

        logfile.write("Product information\n")
        logfile.write("-------------------\n")
        logfile.write(f"Product ID       : {paths.product_id}\n")
        logfile.write(f"Input file       : {events_list}\n")
        logfile.write(f"Output file      : {paths.corrected_events_list}\n")

        logfile.write("\n")

        logfile.write("Run information\n")
        logfile.write("---------------\n")
        logfile.write(f"Resolvit version : {__version__}\n")
        logfile.write(f"Run time         : " f"{datetime.utcnow().isoformat()} UTC\n")

        logfile.write("\n")

        logfile.write("Algorithm parameters\n")
        logfile.write("--------------------\n")
        logfile.write(f"Bin size         : {bin_size}\n")
        logfile.write(f"Event fraction   : " f"{total_events_fraction}\n")
        logfile.write(f"Iterations       : " f"{len(iteration_offsets)}\n")

        for i, offset in enumerate(
            iteration_offsets,
            start=1,
        ):
            logfile.write(f"Iteration {i} offset : " f"{offset}\n")


class ResolvitPaths:
    def __init__(self, events_list, output_dir):
        self.events_list = Path(events_list)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.product_id = self.events_list.name.replace(
            "_l2ce.fits",
            "",
        )

        self.diagnostics_dir = self.output_dir / "diagnostics" / self.product_id

        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        self.corrected_events_list = self.output_dir / self.events_list.name

    def correlation_plot(self, iteration_no, bin_size, t_mid):
        return (
            self.diagnostics_dir
            / f"{iteration_no}_{bin_size:.0f}_{t_mid:.0f}_correlations.png"
        )

    def residual_plot(self, iteration_no):
        return self.diagnostics_dir / f"residuals_iteration_{iteration_no}.png"

    def residual_file(self, iteration_no):
        return self.diagnostics_dir / f"residuals_iteration_{iteration_no}.txt"

    @property
    def log_file(self):
        return self.diagnostics_dir / "resolvit.log"

    @property
    def detector_image(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits",
            "I_l2img.fits",
        )

    @property
    def detector_error(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits",
            "I_l2err.fits",
        )

    @property
    def astronomical_image(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits",
            "A_l2img.fits",
        )

    @property
    def astronomical_error(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits",
            "A_l2err.fits",
        )

    @property
    def detector_exposure(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits",
            "I_l2exp.fits",
        )

    @property
    def astronomical_exposure(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits",
            "A_l2exp.fits",
        )


def read_and_filter_events(events_list_hdu):
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
        self.upper_limit = DEFAULT_UPPER_LIMIT

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

    def get_shifts(self, bin_size=1, correlation_plot=None):
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

    with fits.open(events_list) as events_list_hdu:
        time, fx, fy, photons = read_and_filter_events(events_list_hdu)

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


def apply_detector_corrections(
    events_list,
    residuals,
    corrected_events_list,
):
    with fits.open(events_list) as events_list_hdu:

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

            mask = (time >= r["t_start"]) & (time < r["t_end"])

            dx_events[mask] = r["dx"] - offset_dx
            dy_events[mask] = r["dy"] - offset_dy

        events_list_hdu[1].data["Fx"] = fx + dx_events
        events_list_hdu[1].data["Fy"] = fy + dy_events

        events_list_hdu.writeto(corrected_events_list, overwrite=True)


def update_world_coordinates(corrected_events_list, detector_exposure):
    with (
        fits.open(corrected_events_list, mode="update") as events_list_hdu,
        fits.open(detector_exposure) as exp_hdu,
    ):

        fx = events_list_hdu[1].data["Fx"]
        fy = events_list_hdu[1].data["Fy"]

        w = WCS(exp_hdu[0].header)

        PC1_1 = exp_hdu[0].header["PC1_1"]
        PC1_2 = exp_hdu[0].header["PC1_2"]
        PC2_1 = exp_hdu[0].header["PC2_1"]
        PC2_2 = exp_hdu[0].header["PC2_2"]

        CDELT1 = exp_hdu[0].header["CDELT1"]
        CDELT2 = exp_hdu[0].header["CDELT2"]

        sky = w.pixel_to_world(fx, fy)

        events_list_hdu[1].data["Sky_RA"] = sky.ra.degree
        events_list_hdu[1].data["Sky_DEC"] = sky.dec.degree

        if CDELT1 > 0:
            PC1_1 = -1 * PC1_1
            PC1_2 = -1 * PC1_2

        if CDELT2 < 0:
            PC2_1 = -1 * PC2_1
            PC2_2 = -1 * PC2_2

        (
            events_list_hdu[1].data["Fx_astronomical"],
            events_list_hdu[1].data["Fy_astronomical"],
        ) = detector_to_astronomical(
            fx,
            fy,
            PC1_1,
            PC1_2,
            PC2_1,
            PC2_2,
        )

        events_list_hdu.flush()


def detector_to_astronomical(fx, fy, PC1_1, PC1_2, PC2_1, PC2_2):
    fx_prime = (PC1_1 * (fx - 2400)) + (PC1_2 * (fy - 2400)) + 2400
    fy_prime = (PC2_1 * (fx - 2400)) + (PC2_2 * (fy - 2400)) + 2400
    return fx_prime, fy_prime


def add_resolvit_keywords(header, source_header):

    header.append(source_header.cards["RESOLVIT"])
    header.append(source_header.cards["RSLV_VER"])
    header.append(source_header.cards["RSLV_BIN"])
    header.append(source_header.cards["RSLV_FR"])
    header.append(source_header.cards["RSLV_ITL"])

    for i in range(1, source_header["RSLV_ITL"] + 1):
        header.append(source_header.cards[f"RSLV_IT{i}"])

    header.insert("RESOLVIT", ("", ""))
    header.insert("RESOLVIT", ("", "Resolvit processing information"))
    header.insert("RESOLVIT", ("", ""))


def generate_image_products(
    events_list,
    exposure_map,
    image_file,
    error_file,
    astronomical=False,
):

    if astronomical:
        fx_keyword = "Fx_astronomical"
        fy_keyword = "Fy_astronomical"
    else:
        fx_keyword = "Fx"
        fy_keyword = "Fy"

    with fits.open(events_list) as events_hdu, fits.open(exposure_map) as exp_hdu:

        framecount_per_sec = events_hdu[0].header["AVGFRMRT"]

        fx = events_hdu[1].data[fx_keyword]
        fy = events_hdu[1].data[fy_keyword]
        photons = events_hdu[1].data["EFFECTIVE_NUM_PHOTONS"]
        bad_flag = events_hdu[1].data["BAD FLAG"]

        mask = photons > 0
        mask = np.logical_and(mask, bad_flag)
        fx = fx[mask]
        fy = fy[mask]
        photons = photons[mask]

        image_header = exp_hdu[0].header.copy()

        add_resolvit_keywords(
            image_header,
            events_hdu[0].header,
        )

        bins = np.arange(-0.5, 4800.5, 1)

        events, _, _ = np.histogram2d(
            fy,
            fx,
            bins=(bins, bins),
            weights=photons,
        )

        counts, _, _ = np.histogram2d(
            fy,
            fx,
            bins=(bins, bins),
        )

        exposure = exp_hdu[0].data

        with np.errstate(divide="ignore", invalid="ignore"):
            cps = events / (exposure * framecount_per_sec)
            cps_error = cps / np.sqrt(counts)

        cps[exposure == 0] = 0
        cps_error[exposure == 0] = 0
        cps_error[counts == 0] = 0

        cps = cps.astype(np.float32)
        cps_error = cps_error.astype(np.float32)

        image_hdu = fits.PrimaryHDU(cps)
        image_hdu.header = image_header
        image_hdu.header["FILENAME"] = image_file.name
        image_hdu.header["FILEDATE"] = datetime.utcnow().isoformat(timespec="seconds")
        image_hdu.header["FILEORIG"] = "Resolvit"
        image_hdu.header["DATATYPE"] = "count-rate image"
        image_hdu.writeto(image_file, overwrite=True)

        error_hdu = fits.PrimaryHDU(cps_error)
        error_hdu.header = image_header
        error_hdu.header["FILENAME"] = error_file.name
        error_hdu.header["FILEDATE"] = datetime.utcnow().isoformat(timespec="seconds")
        error_hdu.header["FILEORIG"] = "Resolvit"
        error_hdu.header["DATATYPE"] = "count-rate error image"
        error_hdu.writeto(error_file, overwrite=True)


def process_observation(
    observation_dir,
    bin_size=DEFAULT_BIN_SIZE,
    iteration_offsets=None,
    total_events_fraction=DEFAULT_TOTAL_EVENTS_FRACTION,
):
    """
    Process all UVIT Level2 events lists within an observation.

    The function searches an observation directory recursively for UVIT Level2
    events lists, estimates and corrects the residual drift in each events list,
    and generates corrected data products.

    Parameters
    ----------
    observation_dir : str or pathlib.Path
        Root directory of the UVIT observation.

    bin_size : float, default=100.0
        Temporal bin size, in seconds, used for residual drift estimation.

    iteration_offsets : sequence of float, optional
        Fractional temporal offsets used for successive correction
        iterations. If omitted, the default sequence
        ``[0, 1/2, 1/4, 1/3]`` is used.

    total_events_fraction : float, default=0.75
        Minimum fraction of the median event count required for a
        temporal bin to be used for residual drift estimation.

    Notes
    -----
    Corrected events lists, detector-coordinate images, astronomical
    images, error maps, exposure maps, and diagnostic products are
    written to a ``resolvit_data_products`` directory within the
    observation.
    """

    if iteration_offsets is None:
        iteration_offsets = DEFAULT_ITERATION_OFFSETS

    events_lists = glob(
        f"{observation_dir}/**/data_products/*_data/**/AS1*l2ce.fits",
        recursive=True,
    )

    for events_list in events_lists:
        events_list_path = Path(events_list)

        output_dir = (
            events_list_path.parents[2]
            / "resolvit_data_products"
            / events_list_path.parent.name
        )

        detector_exposure = events_list_path.with_name(
            events_list_path.name.replace("_l2ce.fits", "I_l2exp.fits")
        )

        astronomical_exposure = events_list_path.with_name(
            events_list_path.name.replace("_l2ce.fits", "A_l2exp.fits")
        )

        paths = process_events_list(
            events_list,
            output_dir,
            bin_size,
            iteration_offsets,
            total_events_fraction,
        )

        generate_products(
            paths,
            detector_exposure,
            astronomical_exposure,
        )


def process_events_list(
    events_list,
    output_dir,
    bin_size,
    iteration_offsets,
    total_events_fraction,
):
    """
    Process a single UVIT Level2 events list.

    The function estimates the residual drift in the supplied events
    list, applies detector-coordinate corrections iteratively, and
    produces a corrected events list.

    Parameters
    ----------
    events_list : str or pathlib.Path
        Input UVIT Level2 events list.

    output_dir : str or pathlib.Path
        Directory where the corrected events list and diagnostic products
        will be written.

    bin_size : float
        Temporal bin size, in seconds.

    iteration_offsets : sequence of float
        Fractional temporal offsets used for successive correction
        iterations.

    total_events_fraction : float
        Minimum fraction of the median event count required for a
        temporal bin to participate in residual drift estimation.

    Returns
    -------
    ResolvitPaths
        Object describing the locations of the generated products.
    """

    print(f"Processing {Path(events_list).name}")

    paths = ResolvitPaths(events_list, output_dir)

    write_processing_log(
        paths,
        events_list,
        bin_size,
        total_events_fraction,
        iteration_offsets,
    )

    current_file = events_list

    for i, frac in enumerate(iteration_offsets, start=1):

        offset = bin_size * frac

        residuals = calculate_residuals(
            current_file,
            bin_size,
            offset,
            total_events_fraction,
            i,
            paths,
        )

        apply_detector_corrections(
            current_file,
            residuals,
            paths.corrected_events_list,
        )

        current_file = paths.corrected_events_list

    with fits.open(paths.corrected_events_list, mode="update") as events_list_hdu:

        header = events_list_hdu[0].header

        header["RESOLVIT"] = (True, "Processed using Resolvit")
        header["RSLV_VER"] = (__version__, "Resolvit version")
        header["RSLV_BIN"] = (bin_size, "Time bin size (s)")
        header["RSLV_FR"] = (
            total_events_fraction,
            "Event fraction threshold",
        )
        header["RSLV_ITL"] = (
            len(iteration_offsets),
            "Number of iterations",
        )

        for i, offset in enumerate(iteration_offsets, start=1):
            header[f"RSLV_IT{i}"] = (
                float(offset),
                f"Iteration {i} offset fraction",
            )

        header.insert("RESOLVIT", ("", ""))
        header.insert(
            "RESOLVIT",
            ("", "Resolvit processing information"),
        )
        header.insert("RESOLVIT", ("", ""))

        events_list_hdu.flush()

    return paths


def generate_products(paths, detector_exposure, astronomical_exposure):

    update_world_coordinates(paths.corrected_events_list, detector_exposure)

    for exposure_map in (detector_exposure, astronomical_exposure):
        if exposure_map.exists():
            copy(exposure_map, paths.output_dir / exposure_map.name)

    generate_image_products(
        paths.corrected_events_list,
        paths.detector_exposure,
        paths.detector_image,
        paths.detector_error,
    )

    generate_image_products(
        paths.corrected_events_list,
        paths.astronomical_exposure,
        paths.astronomical_image,
        paths.astronomical_error,
        astronomical=True,
    )

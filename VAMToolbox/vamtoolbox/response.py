# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the GNU GPLv3 license.

import time
import warnings
from cmath import isnan

import matplotlib.pyplot as plt
import numpy as np


class ResponseModel:

    _default_gen_log_fun = {"A": 0, "K": 1, "B": 25, "M": 0.5, "nu": 1}
    _default_linear = {"M": 1, "C": 0}
    _default_interpolation = {"interp_min": 0, "interp_max": 1, "n_pts": 512}

    def __init__(
        self, type: str = "interpolation", form: str = "gen_log_fun", **kwargs
    ):
        """
        Parameters
        ----------
        type : str ('analytical', 'interpolation')
            Select analytical function evaluation or interpolate on pre-built interpolant arrays.
            Interpolation method handles edge cases of input explicitly and hence is more robust.

        form : str ('gen_log_fun', 'linear', 'identity', 'freeform')

        A : float, optional
            parameter in generalized logistic function (Richard's curve)
            Left asymptote

        K : float, optional
            parameter in generalized logistic function (Richard's curve)
            Right asymptote

        B : float, optional
            parameter in generalized logistic function (Richard's curve)
            Steepness of the curve

        M : float, optional
            parameter in generalized logistic function (Richard's curve)
            M shifts the curve left or right. It is the location of inflextion point when nu = 1.

        nu : float, optional
            parameter in generalized logistic function (Richard's curve)
            Influence location of maximum slope relative to the two asymptotes. 'Skew' the curve towards either end.

        M : float, optional
            parameter in linear (affine) function
            M is the slope of the curve: map = M*f + C

        C : float, optional
            parameter in linear (affine) function
            M is the y-intercept of the curve: map = M*f + C

        """
        self.type = type
        self.form = form

        if self.type == "analytical":
            if self.form == "gen_log_fun":
                self.map = self._map_glf
                self.dmapdf = self._dmapdf_glf
                self.map_inv = self._map_inv_glf
                # Shallow copy avoid editing dict '_default_gen_log_fun' in place
                self.params = self._default_gen_log_fun.copy()
                # up-to-date parameters. Default dict is not updated
                self.params.update(kwargs)

            elif self.form == "linear":
                self.map = self._map_lin
                self.dmapdf = self._dmapdf_lin
                self.map_inv = self._map_inv_lin
                # Shallow copy avoid editing dict '_default_linear' in place
                self.params = self._default_linear.copy()  # type: ignore
                # up-to-date parameters. Default dict is not updated
                self.params.update(kwargs)

            elif self.form == "identity":
                self.map = self._map_id
                self.dmapdf = self._dmapdf_id
                self.map_inv = self._map_inv_id

            else:
                raise Exception(
                    "Form: Other analytical functions are not supported yet."
                )

        elif self.type == "interpolation":
            # Interpolation method stores three 1-D arrays as interpolant and query them upon each mapping call.
            # Stored arrays : (1)Sampling point on f, (2)corresponding forward map values, and (3)the first derviative of forward map.
            # All arrays are of the same size. The inverse mapping use (1) and (2) for memory efficiency and avoid singularity problem at asymptotes.

            # function alias
            self.map = self._map_interp
            self.dmapdf = self._dmapdf_interp  # type: ignore
            # Inverse mapping uses the same set of data generated for forward mapping.
            self.map_inv = self._map_inv_interp
            # Shallow copy avoid editing dict '_default_interpolation' in place
            self.params = self._default_interpolation.copy()  # type: ignore

            # build or import interpolant dataset
            if self.form == "gen_log_fun":
                self.params.update(self._default_gen_log_fun)  # Add relevant parameters
                self.params.update(
                    kwargs
                )  # up-to-date parameters. Default dict is not updated

                # build interpolant arrays
                self.interp_f_0 = np.linspace(  # type: ignore
                    self.params["interp_min"],
                    self.params["interp_max"],
                    self.params["n_pts"],
                )
                self.interp_map_0 = self._map_glf(self.interp_f_0)
                self.interp_dmapdf_0 = self._dmapdf_glf(self.interp_f_0)

            elif self.form == "linear":
                self.params.update(self._default_linear)  # Add relevant parameters
                self.params.update(
                    kwargs
                )  # up-to-date parameters. Default dict is not updated

                # build interpolant arrays
                self.interp_f_0 = np.linspace(  # type: ignore
                    self.params["interp_min"],
                    self.params["interp_max"],
                    self.params["n_pts"],
                )
                self.interp_map_0 = self._map_lin(self.interp_f_0)
                self.interp_dmapdf_0 = self._dmapdf_lin(self.interp_f_0)

            elif self.form == "identity":
                self.params.update(
                    kwargs
                )  # up-to-date parameters. Default dict is not updated

                # build interpolant arrays
                self.interp_f_0 = np.linspace(  # type: ignore
                    self.params["interp_min"],
                    self.params["interp_max"],
                    self.params["n_pts"],
                )
                self.interp_map_0 = self._map_id(self.interp_f_0)
                self.interp_dmapdf_0 = self._dmapdf_id(self.interp_f_0)

            elif self.form == "freeform":  # Directly import data instead of generating.
                self.interp_f_0 = kwargs.get(
                    "interp_f_0", None
                )  # Input data points are designated with 0 subscript
                self.interp_map_0 = kwargs.get(
                    "interp_map_0", None
                )  # Input data points are designated with 0 subscript

                # Check inputs
                if (len(self.interp_f_0.shape) > 1) or (
                    len(self.interp_map_0.shape) > 1
                ):
                    raise Exception(
                        'Imported data for material response curve should be 1D. Check "interp_f_0" and "interp_map_0".'
                    )
                if (self.interp_f_0.shape) != (self.interp_map_0.shape):
                    raise Exception(
                        'Size mismatch between "interp_f_0" and "interp_map_0".'
                    )

                # Extending the diff curve by assuming continuity of 1st derivative at the end of the curve
                self.interp_dmapdf_0 = np.diff(
                    self.interp_map_0,
                    n=1,
                    append=(
                        self.interp_map_0[-1]
                        + (self.interp_map_0[-1] - self.interp_map_0[-2])
                    ),
                )
                # Alternative solution to the differed array size is simply using shorter arrays.

            else:
                raise Exception("Other interpolation functions are not supported yet.")

        else:
            raise Exception(
                'Mapping type ("type") should be either "analytical" or "interpolation".'
            )

    # =================================Analytic: Generalized logistic function================================================

    # Definition of generalized logistic function: https://en.wikipedia.org/wiki/Generalised_logistic_function
    def _map_glf(self, f: np.ndarray) -> np.ndarray:
        numerator = self.params["K"] - self.params["A"]

        self.cached_exp = np.exp(
            -self.params["B"] * (f - self.params["M"])
        )  # cache result for later computation of derivative
        denominator = (1 + self.cached_exp) ** (1 / self.params["nu"])

        self.cached_map = self.params["A"] + (
            numerator / denominator
        )  # cache result for later use
        return self.cached_map

    def _dmapdf_glf(self, f: np.ndarray, use_cached_result: bool = False) -> np.ndarray:
        # This function allows pre-computed results to be used to avoid duplicated computations
        # If 'map' is already executed for the exact current input f, use of cached results avoid recomputing the forward map in derivative evaluation.

        coef_1 = (1 / (self.params["K"] - self.params["A"])) ** self.params["nu"]
        coef_2 = self.params["B"] / self.params["nu"]
        if use_cached_result:
            coef_3 = (self.cached_map - self.params["A"]) ** (self.params["nu"] + 1)
            exponential = self.cached_exp
        else:
            coef_3 = (self._map_glf(f) - self.params["A"]) ** (self.params["nu"] + 1)
            exponential = np.exp(-self.params["B"] * (f - self.params["M"]))

        self.cached_dmapdf = coef_1 * coef_2 * coef_3 * exponential
        return self.cached_dmapdf

    def _map_inv_glf(self, mapped: np.ndarray) -> np.ndarray:

        numerator = -np.log(
            ((self.params["K"] - self.params["A"]) / (mapped - self.params["A"]))
            ** self.params["nu"]
            - 1
        )  # Given C=1 and Q=1 --> log(Q)=log(1)=0
        f = (numerator / self.params["B"]) + self.params["M"]

        return f

    # =================================Analytic: Linear (affine) function=====================================================
    # Definition of linear function: mapped = M*f + C
    def _map_lin(self, f: np.ndarray) -> np.ndarray:
        self.cached_map = self.params["M"] * f + self.params["C"]
        return self.cached_map

    def _dmapdf_lin(self, f: np.ndarray, use_cached_result: bool = False) -> np.ndarray:
        return np.ones_like(f) * self.params["M"]

    def _map_inv_lin(self, mapped: np.ndarray) -> np.ndarray:
        return (mapped - self.params["C"]) / self.params["M"]

    # =================================Analytic: Identity function============================================================
    # Definition of identity: mapped = f
    def _map_id(self, f: np.ndarray) -> np.ndarray:
        self.cached_map = f
        return self.cached_map

    def _dmapdf_id(self, f: np.ndarray, use_cached_result: bool = False) -> np.ndarray:
        return np.ones_like(f)

    def _map_inv_id(self, mapped: np.ndarray) -> np.ndarray:
        return mapped

    # =================================Interpolation==========================================================================
    def _map_interp(self, f: np.ndarray) -> np.ndarray:
        """
        Map optical dose to response via interpolation.
        More robust for asymptote values and potentially faster than computing exponentials in generalized logistic function.
        """
        return np.interp(
            f,
            self.interp_f_0,
            self.interp_map_0,
            left=self.interp_map_0[0],
            right=self.interp_map_0[-1],
        )  # Extrapolation points are taken as nearest neighbor, same as default

    def _dmapdf_interp(self, f: np.ndarray) -> np.ndarray:
        """
        Map optical dose to response 1st derivative via interpolation.
        """
        return np.interp(
            f,
            self.interp_f_0,
            self.interp_dmapdf_0,
            left=self.interp_dmapdf_0[0],
            right=self.interp_dmapdf_0[-1],
        )  # Extrapolation points are taken as nearest neighbor, same as default

    def _map_inv_interp(self, mapped: np.ndarray) -> np.ndarray:
        """
        Map material response back to optical dose via interpolation.
        """
        return np.interp(
            mapped,
            self.interp_map_0,
            self.interp_f_0,
            left=self.interp_f_0[0],
            right=self.interp_f_0[-1],
        )  # Extrapolation points are taken as nearest neighbor, same as default

    # =================================Utilities==========================================================================
    def plotMap(
        self, fig=None, ax=None, lb=0, ub=1, n_pts=512, block=True, **plot_kwargs
    ):

        f_test = np.linspace(lb, ub, n_pts)
        mapped_f_test = self.map(f_test)

        if ax == None:
            fig, ax = plt.subplots()

        ax.plot(f_test, mapped_f_test, **plot_kwargs)
        ax.set_xlabel("Optical dose")
        ax.set_ylabel("Material response (mapped dose)")

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            if "label" in plot_kwargs:
                ax.legend()
            plt.show(block=True)

        return fig, ax

    def plotDmapDf(
        self, fig=None, ax=None, lb=0, ub=1, n_pts=512, block=True, **plot_kwargs
    ):

        f_test = np.linspace(lb, ub, n_pts)
        mapped_f_test = self.dmapdf(f_test)

        if ax == None:
            fig, ax = plt.subplots()

        ax.plot(f_test, mapped_f_test, **plot_kwargs)
        ax.set_xlabel("Optical dose")
        ax.set_ylabel("1st derivative of material response (mapped dose)")

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            if "label" in plot_kwargs:
                ax.legend()
            plt.show(block=True)

        return fig, ax

    def plotMapInv(
        self, fig=None, ax=None, lb=0, ub=1, n_pts=512, block=True, **plot_kwargs
    ):

        map_test = np.linspace(lb, ub, n_pts)
        f_test = self.map_inv(map_test)

        if ax == None:
            fig, ax = plt.subplots()

        ax.plot(map_test, f_test, **plot_kwargs)
        ax.set_xlabel("Material response (mapped dose)")
        ax.set_ylabel("Optical dose")

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            if "label" in plot_kwargs:
                ax.legend()
            plt.show(block=True)

        return fig, ax

    def checkResponseTarget(self, f_T: np.ndarray):
        # Check if the response target is reachable with non-negative real inputs, and if it contains inf or nan.
        # Get target range
        f_T_min = np.amin(f_T)
        f_T_max = np.amax(f_T)

        validity = True

        # Check upper limit of response function (only for logistic function)
        if self.form == "gen_log_fun":
            if f_T_max > self.params["K"]:
                warnings.warn(
                    "Maximum response target is greater than right asymptotic value of response function."
                )
                validity = False

        # Check lower limit of response function (for all functional forms), up to 1% tolerance
        if (f_T_min < self.map(0)) and ~(np.isclose(f_T_min, self.map(0), atol=0.01 * f_T_max)):  # type: ignore
            warnings.warn(
                "Minimum response target is lower than response at zero optical dose."
            )
            validity = False

        # Check for boundedness
        if np.isinf(f_T).any():
            warnings.warn("Response target contains infinite value(s).")
            validity = False

        # Check for numeric values
        if np.isnan(f_T).any():
            warnings.warn("Response target contains nan value(s).")
            validity = False

        return validity

    def __repr__(self):
        return str(self.params)

import matplotlib.pyplot as plt

import vamtoolbox


# TODO: make options a dataclass
class Options:

    __default_FBP = {"offset": False}
    __default_CAL = {
        "learning_rate": 0.01,
        "momentum": 0,
        "positivity": 0,
        "sigmoid": 0.01,
    }
    __default_PM = {"rho_1": 1, "rho_2": 1, "p": 1}
    __default_OSMO = {"inhibition": 0}
    __default_BCLP = {
        "response_model": "default",
        "eps": 0.1,
        "weight": 1,
        "p": 2,
        "q": 1,
        "learning_rate": 0.01,
        "optim_alg": "grad_des",
        "g0": None,
    }

    def __init__(
        self,
        method: str = "CAL",
        n_iter: int = 50,
        d_h: float = 0.8,
        d_l: float = 0.7,
        filter: str = "ram-lak",
        units: str = "normalized",
        blb=0,
        bub=None,
        **kwargs
    ):
        """
        Parameters
        ----------

        method : str
            Type of VAM method
                - "FBP"
                - "CAL"
                - "PM"
                - "OSMO"
                - "BCLP"

        n_iter : int
            number of iterations to perform

        d_h : float
            in-target dose constraint

        d_l : float
            out-of-target dose constraint

        filter : str
            filter for initialization ("ram-lak", "shepp-logan", "cosine", "hamming", "hanning", None)

        blb : double
            lower bound of sinogram pixel value. If None is given, no lower limit.

        bub : double
            upper bound of sinogram pixel value. If None is given, no upper limit.

        learning_rate : float, optional (CAL) (BCLP)
            step size in approximate gradient descent

        momentum : float, optional (CAL)
            descent momentum for faster convergence

        positivity : float, optional (CAL)
            positivity constraint enforced at each iteration

        sigmoid : float, optional (CAL)
            sigmoid thresholding strength

        rho_1 : float, optional (PM)

        rho_2 : float, optional (PM)

        p : int, optional (PM)

        inhibition : float, optional (OSMO)

        response model : ResponseModel, optional (BCLP)
            ResponseModel object to capture material response

        eps : float, optional (BCLP)
            band tolerance is +-eps around target value. Scalar or same array size as target array.

        weights : float, optional (BCLP)
            weightings in Lp minimization. Scalar (no local emphasis) or same array size as target array.

        p : float, optional (BCLP)
            p as in Lp norm. Not required to be integer in BCLP.

        q : float, optional (BCLP)
            Cost function is a Lp norm (scalar) raised to q-th power. Changing q does not affect minimizer on solution landscape but affect convergence behavior.

        g0 : sinogram, optional (BCLP)
            Initial guess of sinogram solution. Can be obtained from saved result or

        """
        self.method = method
        self.n_iter = n_iter
        self.d_h = d_h
        self.d_l = d_l
        self.filter = filter
        self.units = units
        self.blb = blb
        self.bub = bub

        self.__default_FBP.update(kwargs)
        self.__default_CAL.update(kwargs)
        self.__default_PM.update(kwargs)
        self.__default_OSMO.update(kwargs)
        self.__default_BCLP.update(
            kwargs
        )  # TODO: The class definition of dict "__default_BCLP" should not be edited in place here.
        self.__dict__.update(kwargs)  # Store all the extra variables

        self.verbose = self.__dict__.get("verbose", False)
        self.save_img_path = self.__dict__.get("save_img_path", None)
        self.bit_depth = self.__dict__.get("bit_depth", None)
        self.exit_param = self.__dict__.get("exit_param", None)

        if method == "FBP":
            self.offset = self.__default_FBP["offset"]

        if method == "CAL":
            self.learning_rate = self.__default_CAL["learning_rate"]
            self.momentum = self.__default_CAL["momentum"]
            self.positivity = self.__default_CAL["positivity"]
            self.sigmoid = self.__default_CAL["sigmoid"]

        if method == "PM":
            self.rho_1 = self.__default_PM["rho_1"]
            self.rho_2 = self.__default_PM["rho_2"]
            self.p = self.__default_PM["p"]

        if method == "OSMO":
            self.inhibition = self.__default_OSMO["inhibition"]

        if method == "BCLP":
            if self.__default_BCLP["response_model"] == "default":
                # Initialize a response model by default, only upon __init__ of Options class.
                # This avoids putting the ResponseModel object inside the class definition of Options (and hence avoid import problems and unnesscary init of default response model)
                self.response_model = vamtoolbox.response.ResponseModel()
            else:
                # If a response model is given, use the provided one instead.
                self.response_model = self.__default_BCLP["response_model"]  # type: ignore

            self.eps = self.__default_BCLP["eps"]
            self.weight = self.__default_BCLP["weight"]
            self.p = self.__default_BCLP["p"]  # type: ignore
            self.q = self.__default_BCLP["q"]  # type: ignore
            self.learning_rate = self.__default_BCLP["learning_rate"]  # type: ignore
            self.optim_alg = self.__default_BCLP["optim_alg"]
            self.g0 = self.__default_BCLP["g0"]
            self.test_alternate_handling = self.__dict__.get(
                "test_alternate_handling", False
            )  # flag for testing alternate handling. Default: False. This will override original weight setting. Will be removed for actual release

    def __str__(self):
        return str(self.__dict__)


def optimize(
    target_geo: vamtoolbox.geometry.TargetGeometry,
    proj_geo: vamtoolbox.geometry.ProjectionGeometry,
    options: Options,
    output="packaged",
):
    """
    Performs VAM optimization using the selected optimizer in options

    Parameters
    ----------
    target_geo : geometry.TargetGeometry object

    proj_geo : geometry.ProjectionGeometry object

    options : optimize.Options object

    Returns
    -------
    geometry.Sinogram object

    geometry.Reconstruction object

    """

    if options.units != "normalized" or proj_geo.absorption_coeff is not None:
        proj_geo.calcAbsorptionMask(target_geo)

    if options.method == "FBP":
        return vamtoolbox.optimizer.FBP.minimizeFBP(target_geo, proj_geo, options)

    elif options.method == "CAL":
        return vamtoolbox.optimizer.CAL.minimizeCAL(target_geo, proj_geo, options)

    elif options.method == "PM":
        return vamtoolbox.optimizer.PM.minimizePM(target_geo, proj_geo, options)

    elif options.method == "OSMO":
        return vamtoolbox.optimizer.OSMO.minimizeOSMO(target_geo, proj_geo, options)

    elif options.method == "BCLP":
        return vamtoolbox.optimizer.BCLP.minimizeBCLP(
            target_geo, proj_geo, options, output
        )

import matplotlib.pyplot as plt

import vamtoolbox


class Options:

    __default_FBP = {"offset":False}
    __default_CAL = {"learning_rate":0.01,"momentum":0,"positivity":0,"sigmoid":0.01}
    __default_PM = {"rho_1":1,"rho_2":1,"p":1}
    __default_OSMO = {"inhibition":0}

    def __init__(self,method : str ='CAL',n_iter : int = 50,d_h : float = 0.8,d_l : float = 0.7,filter : str ='ram-lak',units:str='normalized',**kwargs):
        """
        Parameters
        ----------

        method : str
            Type of VAM method
                - "FBP"
                - "CAL"
                - "PM"
                - "OSMO"
        
        n_iter : int
            number of iterations to perform

        d_h : float
            in-target dose constraint

        d_l : float
            out-of-target dose constraint

        filter : str
            filter for initialization ("ram-lak", "shepp-logan", "cosine", "hamming", "hanning", None)

        learning_rate : float, optional (CAL)
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




        """
        self.method = method
        self.n_iter = n_iter
        self.d_h = d_h
        self.d_l = d_l
        self.filter = filter
        self.units = units
        self.__default_FBP.update(kwargs)
        self.__default_CAL.update(kwargs)
        self.__default_PM.update(kwargs)
        self.__default_OSMO.update(kwargs)
        self.__dict__.update(kwargs)  # Store all the extra variables

        self.verbose = self.__dict__.get('verbose',False)
        self.bit_depth = self.__dict__.get('bit_depth',None)
        self.exit_param = self.__dict__.get('exit_param',None)

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




def optimize(target_geo : vamtoolbox.geometry.TargetGeometry,proj_geo : vamtoolbox.geometry.ProjectionGeometry,options:Options):
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
        return vamtoolbox.optimizer.FBP.minimizeFBP(target_geo,proj_geo,options)

    elif options.method == "CAL":
        return vamtoolbox.optimizer.CAL.minimizeCAL(target_geo,proj_geo,options)

    elif options.method == "PM":
        return vamtoolbox.optimizer.PM.minimizePM(target_geo,proj_geo,options)

    elif options.method == "OSMO":
        return vamtoolbox.optimizer.OSMO.minimizeOSMO(target_geo,proj_geo,options)


        






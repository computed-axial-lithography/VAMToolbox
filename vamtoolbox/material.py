import numpy as np
import matplotlib.pyplot as plt


class ResponseModel:

    __default_gen_log_fun = {"A": 0, "K": 1, "B": 25, "M": 0.5, "nu": 1}
    __default_linear = {"M":1, "C":0} 

    def __init__(self,type : str = "analytical", form :str = "gen_log_fun", **kwargs):
        """
        Parameters
        ----------
        type : str ("analytical", "numerical")

        form : str ("gen_log_fun")

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
            Influence location of maximum slope relative to the two asymptotes. "Skew" the curve towards either end.

        """

        if type == "analytical":
            if form == "gen_log_fun":
                self.map = self.__map_glf__
                self.dmapdf = self.__dmapdf_glf__
                self.map_inv = self.__map_inv_glf__
                self.params = self.__default_gen_log_fun.copy() #Shallow copy avoid editing dict "__default_gen_log_fun" in place 
                self.params.update(kwargs) #up-to-date parameters. Default dict is not updated
                
            elif form == "linear":  
                self.map = self.__map_lin__
                self.dmapdf = self.__dmapdf_lin__
                self.map_inv = self.__map_inv_lin__
                self.params = self.__default_linear.copy() #Shallow copy avoid editing dict "__default_gen_log_fun" in place 
                self.params.update(kwargs) #up-to-date parameters. Default dict is not updated

            elif form == "identity":
                self.map = self.__map_id__
                self.dmapdf = self.__dmapdf_id__
                self.map_inv = self.__map_inv_id__

            else:
                raise Exception("Other analytical function is not supported yet.")

        else:
            raise Exception("Numerical mapping is not supported yet.")


    #Definition of generalized logistic function: https://en.wikipedia.org/wiki/Generalised_logistic_function
    def __map_glf__(self, f : np.ndarray):
        numerator = self.params["K"] - self.params["A"]

        self.cached_exp = np.exp(-self.params["B"]*(f-self.params["M"])) #cache result for later computation of derivative
        denominator = (1+self.cached_exp)**(1/self.params["nu"])
        
        self.cached_map = self.params["A"] + (numerator/denominator)  #cache result for later use
        return self.cached_map


    def __dmapdf_glf__(self, f : np.ndarray, use_cached_result : bool = False):
        #This function allows pre-computed results to be used to avoid duplicated computations

        coef_1 = ((1/(self.params["K"] - self.params["A"]))**self.params["nu"])
        coef_2 = (self.params["B"]/self.params["nu"])
        if use_cached_result:
            coef_3 = (self.cached_map-self.params["A"])**(self.params["nu"]+1)
            exponential = self.cached_exp
        else:
            coef_3 = (self.__map_glf__(f)-self.params["A"])**(self.params["nu"]+1)
            exponential = np.exp(-self.params["B"]*(f-self.params["M"]))

        self.cached_dmapdf = coef_1*coef_2*coef_3*exponential 
        return self.cached_dmapdf

    def __map_inv_glf__(self, mapped : np.ndarray):
        
        numerator = -np.log(((self.params["K"] - self.params["A"])/(mapped - self.params["A"]))**self.params["nu"] - 1) #Given C=1 and Q=1 --> log(Q)=log(1)=0
        f = (numerator/self.params["B"]) + self.params["M"]

        return f

    #Definition of linear function: mapped = M*f + C
    def __map_lin__(self, f : np.ndarray):
        self.cached_map = self.params["M"]*f + self.params["C"]
        return self.cached_map

    def __dmapdf_lin__(self, f : np.ndarray, use_cached_result : bool = False):
        return np.ones_like(f)*self.params["M"]

    def __map_inv_lin__(self, mapped : np.ndarray):
        return (mapped-self.params["C"])/self.params["M"]

    #Definition of identity: mapped = f
    def __map_id__(self, f : np.ndarray):
        self.cached_map = f
        return self.cached_map

    def __dmapdf_id__(self, f : np.ndarray, use_cached_result : bool = False):
        return np.ones_like(f)

    def __map_inv_id__(self, mapped : np.ndarray):
        return mapped



    def plotMap(self, lb = 0, ub = 1, n_pts=100, block=False, show = True):

        f_test = np.linspace(lb,ub,n_pts)
        mapped_f_test = self.map(f_test)
        # plt.figure()
        plt.plot(f_test, mapped_f_test)
        if block == False:
            plt.ion()
        if show == True:
            plt.show()


    def plotDmapDf(self, lb = 0, ub = 1, n_pts=100, block=False, show = True):

        f_test = np.linspace(lb,ub,n_pts)
        mapped_f_test = self.dmapdf(f_test)
        # plt.figure()
        plt.plot(f_test, mapped_f_test)
        if block == True:
            plt.ioff()
        else:
            plt.ion()

        if show == True:
            plt.show()    



    def checkResponseTarget(self, f_T):
        #Check if the response target is reachable with non-negative real inputs.
        f_T_min = np.amin(f_T)
        f_T_max = np.amax(f_T)

        try:
            #Inversion
            #All inverted inputs within non-negative real?
            pass
        except:
            #Inversion error
            pass


## Test

if __name__ == "__main__":

    plt.figure()
    plt.title('Generalized logistic function with varying B')
    test_rm = ResponseModel(B=10)
    test_rm.plotMap(block=False, show = False)
    print(test_rm.params)

    test_rm = ResponseModel(B=25)
    test_rm.plotMap(block=False, show = False)
    print(test_rm.params)

    test_rm = ResponseModel(B=50)
    test_rm.plotMap(block=False, show = False)
    print(test_rm.params)

    plt.figure()
    plt.title('Generalized logistic function with varying nu')
    test_rm = ResponseModel(nu=0.2)
    test_rm.plotMap(block=False, show = False)
    print(test_rm.params)

    test_rm = ResponseModel(nu=1)
    test_rm.plotMap(block=False, show = False)
    print(test_rm.params)

    test_rm = ResponseModel(nu=5)
    test_rm.plotMap(block=False, show = False)
    print(test_rm.params)

    plt.figure()
    plt.title('Derivative of generalized logistic function with varying B')
    test_rm = ResponseModel(B=10)
    test_rm.plotDmapDf(block=False, show = False)
    print(test_rm.params)

    test_rm = ResponseModel(B=25)
    test_rm.plotDmapDf(block=False, show = False)
    print(test_rm.params)

    test_rm = ResponseModel(B=50)
    test_rm.plotDmapDf(block=False, show = False)
    print(test_rm.params)

    plt.figure()
    plt.title('Derivative of generalized logistic function with varying nu')
    test_rm = ResponseModel(nu=0.2)
    test_rm.plotDmapDf(block=False, show = False)
    print(test_rm.params)

    test_rm = ResponseModel(nu=1)
    test_rm.plotDmapDf(block=False, show = False)
    print(test_rm.params)

    test_rm = ResponseModel(nu=5)
    test_rm.plotDmapDf(block=False, show = False)
    print(test_rm.params)

    #Check inversion function


    plt.show()
    input()

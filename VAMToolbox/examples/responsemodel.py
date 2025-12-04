import vamtoolbox as vam
import matplotlib.pyplot as plt

# the response model maps optical dose to material response
# it is defined generally with a parameterized logistic function
# following examples show how the function shape varies with some of these parameters
# see the documentation for ResponseModel for detailed description

# varying B changes the slope of the logistic function
fig, axs = plt.subplots(1, 1)
test_rm = vam.response.ResponseModel(form="linear")
test_rm.plotMap(fig, axs, label="linear", block=False)

test_rm = vam.response.ResponseModel(form="gen_log_fun", B=10)
test_rm.plotMap(fig, axs, label="B=10", block=False)

test_rm = vam.response.ResponseModel(form="gen_log_fun", B=25)
test_rm.plotMap(fig, axs, label="B=25", block=False)

test_rm = vam.response.ResponseModel(form="gen_log_fun", B=50)
test_rm.plotMap(fig, axs, label="B=50", block=True)


# varying nu 'skews' the inflection point towards either end
fig, axs = plt.subplots(1, 1)
test_rm = vam.response.ResponseModel(form="gen_log_fun", nu=0.2)
test_rm.plotMap(fig, axs, label="nu=0.2", block=False)

test_rm = vam.response.ResponseModel(form="gen_log_fun", nu=1)
test_rm.plotMap(fig, axs, label="nu=1", block=False)

test_rm = vam.response.ResponseModel(form="gen_log_fun", nu=5)
test_rm.plotMap(fig, axs, label="nu=5", block=True)

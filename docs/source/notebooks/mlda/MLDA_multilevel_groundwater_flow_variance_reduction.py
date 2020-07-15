# Import modules

# Import groundwater flow model utils
import os
import numpy as np
import time
import pymc3 as pm
import theano.tensor as tt
from itertools import product
import matplotlib.pyplot as plt
from Model import Model, model_wrapper, project_eigenpairs

os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Set environmental variable

# Set parameters

# Set the resolution of the multi-level models (from coarsest to finest)
# This is a list of different model resolutions. Each
# resolution added to the list will add one level to the multi-level
# inference. Each element is a tuple (x,y) where x, y are the number of
# points in each dimension. For example, here we set resolutions =
# [(30, 30), (120, 120)] which creates a coarse, cheap 30x30 model and
# a fine, expensive 120x120 model.
resolutions = [(10, 10), (20, 20), (40, 40)]

# Set random field parameters
field_mean = 0
field_stdev = 1
lamb_cov = 0.1

# Set the number of unknown parameters (i.e. dimension of theta in posterior)
nparam = 2

# Number of draws from the distribution
ndraws = 500

# Number of burn-in samples
nburn = 200

# MLDA and Metropolis tuning parameters
tune = True
tune_interval = 100
discard_tuning = True

# Number of independent chains
nchains = 20

# Subsampling rate for MLDA
nsub = 3

# variance reduction
vr = True

# Do blocked/compounds sampling in Metropolis and MLDA
# Note: This choice applies only to the coarsest level in MLDA
# (where a Metropolis sampler is used), all other levels use block sampling
blocked = False

# Set the sigma for inference
sigma = 0.01

# Data generation seed
data_seed = 12345

# Sampling seed
sampling_seed = 12345

# Datapoints list
points_list = [0.1, 0.3, 0.5, 0.7, 0.9]


# Define a function that calculates the quantity of interest Q given the model
# It needs to be applied after feeding the model with a theta a doing the solve
# In this case, the quantity of interest is the hydraulic head at some fixed point (x, y)=(0.5, 0.45)
def quantity_of_interest(my_model):
    """Quantity of interest function"""
    return my_model.solver.h(0.5, 0.45)


# Use a Theano Op along with the code within ./mlda to construct the forward model
class ForwardModel(tt.Op):
    """
    Theano Op that wraps the forward model computation,
    necessary to pass "black-box" fenics code into pymc3.
    Based on the work in:
    https://docs.pymc.io/notebooks/blackbox_external_likelihood.html
    https://docs.pymc.io/Advanced_usage_of_Theano_in_PyMC3.html
    """

    # Specify what type of object will be passed and returned to the Op when it is
    # called. In our case we will be passing it a vector of values (the parameters
    # that define our model) and returning a a vector of model outputs
    itypes = [tt.dvector]  # expects a vector of parameter values (theta)
    otypes = [tt.dvector]  # outputs a vector of model outputs

    def __init__(self, my_model, x, pymc3_model):
        """
        Initialise the Op with various things that our forward model function
        requires.
        Parameters
        ----------
        my_model:
            A Model object (defined in file model.py) that contains the parameters
            and functions of our model.
        x:
            The dependent variable (aka 'x') that our model requires. This is
            the datapoints in this example.
        """
        # add inputs as class attributes
        self.my_model = my_model
        self.x = x
        self.pymc3_model = pymc3_model

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta = inputs[0]  # this will contain my variables

        # call the forward model function
        outputs[0][0] = model_wrapper(self.my_model, theta, self.x)

        # call the quantity of interest function after the model has been solved
        # with the provided theta
        # save the result inside the pymc3 model variable Q
        with self.pymc3_model:
            pm.set_data({"Q": quantity_of_interest(self.my_model)})


# Instantiate Model objects and data

# Note this can take several minutes for large resolutions
my_models = []
for r in resolutions:
    my_models.append(Model(r, field_mean, field_stdev, nparam, lamb_cov))

# Project eignevactors from fine model to all coarse models
for i in range(len(my_models[:-1])):
    project_eigenpairs(my_models[-1], my_models[i])

# Solve finest model as a test and plot transmissivity field and solution
np.random.seed(data_seed)
my_models[-1].solve()
my_models[-1].plot(lognormal=False)

# Save true parameters of finest model
true_parameters = my_models[-1].random_process.parameters

# Define the sampling points.
x_data = y_data = np.array(points_list)
datapoints = np.array(list(product(x_data, y_data)))

# Get data from the sampling points and perturb it with some noise.
noise = np.random.normal(0, 0.001, len(datapoints))

# Generate data from the finest model for use in pymc3 inference - these data are used in all levels
data = model_wrapper(my_models[-1], true_parameters, datapoints) + noise

print("True Q: " + str(quantity_of_interest(my_models[-1])))

# Create covariance matrix of normal error - it is a diagonal matrix
s = np.identity(len(data))
np.fill_diagonal(s, sigma ** 2)

# Instantiate forward model objects

# create Theano Ops to wrap likelihoods of all model levels and store them in list
mout = []

# Construct pymc3 model objects for coarse models

# Set up models in pymc3 for each level - excluding finest model level
coarse_models = []
for j in range(len(my_models) - 1):
    with pm.Model() as model:
        # A variable Q has to be defined if you want to use the variance reduction feature
        # Q can be of any dimension - here it a scalar
        Q = pm.Data('Q', np.float64(0.0))

        # Sigma_e is the covariance of the assumed error 'e' in the model.
        # This error is due to measurement noise/bias vs. the real world
        Sigma_e = pm.Data('Sigma_e', s)

        # uniform priors on unknown parameters
        parameters = []
        for i in range(nparam):
            parameters.append(pm.Uniform('theta_' + str(i), lower=-3., upper=3.))

        # convert thetas to a tensor vector
        theta = tt.as_tensor_variable(parameters)

        # this is a deterministic variable that captures the output of
        # the forward model every time it is run
        mout.append(ForwardModel(my_models[j], datapoints, model))
        output = pm.Potential('output', mout[j](theta))

        # The distribution of the error 'e' (assumed error of the forward model)
        # This is multi-variate normal.
        # This creates the likelihood of the model given the observed data
        pm.MvNormal('e', mu=output, cov=Sigma_e, observed=data)

    coarse_models.append(model)

# Perform inference using MLDA and Metropolis

# Set up finest model and perform inference with PyMC3, using the MLDA algorithm
# and passing the coarse_models list created above.
method_names = []
traces = []
runtimes = []
acc = []
ess = []
ess_n = []
performances = []

with pm.Model() as fine_model:
    # A variable Q has to be defined if you want to use the variance reduction feature
    # Q can be of any dimension - here it a scalar
    Q = pm.Data('Q', np.float64(0.0))

    # Sigma_e is the covariance of the assumed error 'e' in the model.
    # This error is due to measurement noise/bias vs. the real world
    Sigma_e = pm.Data('Sigma_e', s)

    # uniform priors on unknown parameters
    parameters = []
    for i in range(nparam):
        parameters.append(pm.Uniform('theta_' + str(i), lower=-3., upper=3.))

    # convert thetas to a tensor vector
    theta = tt.as_tensor_variable(parameters)

    # this is a deterministic variable that captures the output of
    # the fine forward model every time it is run
    mout.append(ForwardModel(my_models[-1], datapoints, fine_model))
    output = pm.Potential('output', mout[-1](theta))

    # The distribution of the error 'e' (assumed error of the forward model)
    pm.MvNormal('e', mu=output, cov=Sigma_e, observed=data)

    # Initialise an MLDA step method object, passing the subsampling rate and
    # coarse models list - notice that we define variance reduction and also we
    # set store_Q_fine to True. This will make sure the sampler store the
    # quantities of interest at the fine level, to allow us to easily compare
    # the accuracy of the Q calculation between using the first chain vs. using all chain
    # differences
    step_mlda_with = pm.MLDA(subsampling_rates=nsub, coarse_models=coarse_models,
                             tune=tune, tune_interval=tune_interval, base_blocked=blocked,
                             variance_reduction=vr, store_Q_fine=True)

    # MLDA with variance reduction
    t_start = time.time()
    method_names.append("MLDA_with_vr")
    traces.append(pm.sample(draws=ndraws, step=step_mlda_with,
                            chains=nchains, tune=nburn,
                            discard_tuned_samples=discard_tuning,
                            random_seed=sampling_seed))
    runtimes.append(time.time() - t_start)

# Print performance metrics
for i, trace in enumerate(traces):
    acc.append(trace.get_sampler_stats('accepted').mean())
    ess.append(np.array(pm.ess(trace).to_array()))
    ess_n.append(ess[i] / len(trace) / trace.nchains)
    performances.append(ess[i] / runtimes[i])
    print(f'\nSampler {method_names[i]}: {len(trace)} drawn samples in each of '
          f'{trace.nchains} chains.'
          f'\nRuntime: {runtimes[i]} seconds'
          f'\nAcceptance rate: {acc[i]}'
          f'\nESS list: {ess[i]}'
          f'\nNormalised ESS list: {ess_n[i]}'
          f'\nESS/sec: {performances[i]}')

# Show stats summary
# Print true theta values and pymc3 sampling summary
print(f"\nDetailed summaries and plots:\nTrue parameters: {true_parameters}")
for i, trace in enumerate(traces):
    print(f"\nSampler {method_names[i]}:\n", pm.stats.summary(trace))

# Show traceplots
# Print true theta values and pymc3 sampling summary
for i, trace in enumerate(traces):
    pm.plots.traceplot(trace)

plt.show()

# Compare variance of Q estimation between:
# - Standard approach: Using only Q values from the fine chain (Q_2)
# - Telescopic sum approach: Using Q values from the the coarsest chain (Q_0),
#   plus all estimates of differences between levels (e.g. Q_1_0, Q_2_1, etc)
Q_2 = traces[0].get_sampler_stats("Q_2").reshape((nchains, ndraws))
Q_0 = np.concatenate(traces[0].get_sampler_stats("Q_0")).reshape((nchains, ndraws * nsub * nsub))
Q_1_0 = np.concatenate(traces[0].get_sampler_stats("Q_1_0")).reshape((nchains, ndraws * nsub))
Q_2_1 = np.concatenate(traces[0].get_sampler_stats("Q_2_1")).reshape((nchains, ndraws))
Q_mean_standard = Q_2.mean(axis=1).mean()
Q_var_standard = Q_2.mean(axis=1).var()
Q_mean_vr = (Q_0.mean(axis=1) + Q_1_0.mean(axis=1) + Q_2_1.mean(axis=1)).mean()
Q_var_vr = (Q_0.mean(axis=1) + Q_1_0.mean(axis=1) + Q_2_1.mean(axis=1)).var()

print(f"Standard method:    Mean: {Q_mean_standard}    Variance: {Q_var_standard}")
print(f"Telescopic method:    Mean: {Q_mean_vr}    Variance: {Q_var_vr}")

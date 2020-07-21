
from pymc3.sampling import assign_step_methods, sample
from pymc3.model import Model, Potential, set_data, Deterministic
from pymc3.step_methods import (
    NUTS,
    BinaryGibbsMetropolis,
    CategoricalGibbsMetropolis,
    Metropolis,
    Slice,
    CompoundStep,
    NormalProposal,
    MultivariateNormalProposal,
    RecursiveDAProposal,
    HamiltonianMC,
    EllipticalSlice,
    DEMetropolis,
    DEMetropolisZ,
    MLDA
)

from pymc3.distributions import Binomial, Normal, MvNormal
from pymc3.data import Data

import numpy as np
import theano.tensor as tt


size = 200
true_intercept = 1
true_slope = 2
sigma = 1

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + np.random.normal(0, sigma**2, size)

#data = dict(x=x, y=y)
s = np.identity(y.shape[0])
np.fill_diagonal(s, sigma ** 2)

class ForwardModel(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, x, pymc3_model):

        self.x = x
        self.pymc3_model = pymc3_model

    def perform(self, node, inputs, outputs):
        intercept = inputs[0][0]
        x_coeff = inputs[0][1]

        temp = intercept + x_coeff * x + self.pymc3_model.bias.get_value() + np.random.normal(0.0, 0.001)
        with self.pymc3_model:
            set_data({'model_output': temp})
        outputs[0][0] = temp


mout = []

coarse_models = []

with Model() as coarse_model_0:
    mu_B = Data('mu_B', np.zeros(y.shape))
    bias = Data('bias', 3.5*np.ones(y.shape))
    Sigma_B = Data('Sigma_B', np.zeros((y.shape[0], y.shape[0])))
    model_output = Data('model_output', np.zeros(y.shape))
    Sigma_e = Data('Sigma_e', s)

    # Define priors
    #sigma = HalfCauchy('sigma', beta=10, testval=1.)
    intercept = Normal('Intercept', 0, sigma=20)
    x_coeff = Normal('x', 0, sigma=20)

    theta = tt.as_tensor_variable([intercept, x_coeff])

    mout.append(ForwardModel(x, coarse_model_0))

    output = Potential('output', mout[0](theta))

    # Define likelihood
    likelihood = MvNormal('y', mu=output + mu_B,
                        cov=Sigma_e, observed=y)

    coarse_models.append(coarse_model_0)

with Model() as coarse_model_1:
    mu_B = Data('mu_B', np.zeros(y.shape))
    bias = Data('bias', 2.2 * np.ones(y.shape))
    Sigma_B = Data('Sigma_B', np.zeros((y.shape[0], y.shape[0])))
    model_output = Data('model_output', np.zeros(y.shape))
    Sigma_e = Data('Sigma_e', s)

    # Define priors
    # sigma = HalfCauchy('sigma', beta=10, testval=1.)
    intercept = Normal('Intercept', 0, sigma=20)
    x_coeff = Normal('x', 0, sigma=20)

    theta = tt.as_tensor_variable([intercept, x_coeff])

    mout.append(ForwardModel(x, coarse_model_1))

    output = Potential('output', mout[1](theta))

    # Define likelihood
    likelihood = MvNormal('y', mu=output + mu_B,
                        cov=Sigma_e, observed=y)

    coarse_models.append(coarse_model_1)

with Model() as model:
    bias = Data('bias', np.zeros(y.shape))
    model_output = Data('model_output', np.zeros(y.shape))
    Sigma_e = Data('Sigma_e', s)

    # Define priors
    # sigma = HalfCauchy('sigma', beta=10, testval=1.)
    intercept = Normal('Intercept', 0, sigma=20)
    x_coeff = Normal('x', 0, sigma=20)

    theta = tt.as_tensor_variable([intercept, x_coeff])

    mout.append(ForwardModel(x, model))

    output = Potential('output', mout[-1](theta))

    # Define likelihood
    likelihood = MvNormal('y', mu=output,
                          cov=Sigma_e, observed=y)

    step_mlda = MLDA(coarse_models=coarse_models,
                     adaptive_error_correction=True)

    trace_mlda = sample(draws=800, step=step_mlda,
                        chains=2, tune=500,
                        discard_tuned_samples=False)

    m0 = step_mlda.next_step_method.next_model.mu_B.get_value()
    s0 = step_mlda.next_step_method.next_model.Sigma_B.get_value()
    m1 = step_mlda.next_model.mu_B.get_value()
    s1 = step_mlda.next_model.Sigma_B.get_value()
    print(f"m0: {m0}")
    print(f"s0: {s0}")
    print(f"m1: {m1}")
    print(f"s1: {s1}")

    #trace_metropolis = sample(draws=1000, step=step_metropolis,
    #                    chains=4, tune=1000)


'''
mout = []

coarse_models = []
with Model() as coarse_model_0:
    mu_B = Data('mu_B', np.zeros(y.shape))
    bias = Data('bias', 0.2*np.ones(y.shape))
    Sigma_B = Data('Sigma_B', np.zeros((y.shape[0], y.shape[0])))
    model_output = Data('model_output', np.zeros(y.shape))
    Sigma_e = Data('Sigma_e', s)

    # Define priors
    #sigma = HalfCauchy('sigma', beta=10, testval=1.)
    intercept = Normal('Intercept', 0, sigma=20)
    x_coeff = Normal('x', 0, sigma=20)

    theta = tt.as_tensor_variable([intercept, x_coeff])

    mout.append(ForwardModel(x, coarse_model_0))

    output = Potential('output', mout[0](theta))

    # Define likelihood
    likelihood = MvNormal('y', mu=output + mu_B,
                        cov=Sigma_e, observed=y)

    coarse_models.append(coarse_model_0)

with Model() as coarse_model_1:
    mu_B = Data('mu_B', np.zeros(y.shape))
    bias = Data('bias', 0.1 * np.ones(y.shape))
    Sigma_B = Data('Sigma_B', np.zeros((y.shape[0], y.shape[0])))
    model_output = Data('model_output', np.zeros(y.shape))
    Sigma_e = Data('Sigma_e', s)

    # Define priors
    # sigma = HalfCauchy('sigma', beta=10, testval=1.)
    intercept = Normal('Intercept', 0, sigma=20)
    x_coeff = Normal('x', 0, sigma=20)

    theta = tt.as_tensor_variable([intercept, x_coeff])

    mout.append(ForwardModel(x, coarse_model_1))

    output = Potential('output', mout[1](theta))

    # Define likelihood
    likelihood = MvNormal('y', mu=output + mu_B,
                          cov=Sigma_e, observed=y)

    coarse_models.append(coarse_model_1)

with Model() as model:
    bias = Data('bias', np.zeros(y.shape))
    model_output = Data('model_output', np.zeros(y.shape))
    Sigma_e = Data('Sigma_e', s)

    # Define priors
    # sigma = HalfCauchy('sigma', beta=10, testval=1.)
    intercept = Normal('Intercept', 0, sigma=20)
    x_coeff = Normal('x', 0, sigma=20)

    theta = tt.as_tensor_variable([intercept, x_coeff])

    mout.append(ForwardModel(x, model))

    output = Potential('output', mout[-1](theta))

    # Define likelihood
    likelihood = MvNormal('y', mu=output,
                          cov=Sigma_e, observed=y)

    step_mlda_2 = MLDA(coarse_models=coarse_models)

    trace_mlda_2 = sample(draws=1000, step=step_mlda_2,
                          chains=1, tune=1000)

    m0 = step_mlda_2.next_step_method.next_model.mu_B.get_value()
    s0 = step_mlda_2.next_step_method.next_model.Sigma_B.get_value()
    m1 = step_mlda_2.next_model.mu_B.get_value()
    s1 = step_mlda_2.next_model.Sigma_B.get_value()
    print(f"m0b: {m0}")
    print(f"s0b: {s0}")
    print(f"m1b: {m1}")
    print(f"s1b: {s1}")

'''
print(trace_mlda.get_sampler_stats('accepted').mean())
#print(trace_mlda_2.get_sampler_stats('accepted').mean())
#print(trace_metropolis.get_sampler_stats('accepted').mean())

from pymc3.stats import summary
from pymc3.plots import traceplot
import matplotlib.pyplot as plt
print(summary(trace_mlda))
#print(summary(trace_mlda_2))
#print(summary(trace_metropolis))
traceplot(trace_mlda)
#traceplot(trace_mlda_2)
#traceplot(trace_metropolis)
plt.show()
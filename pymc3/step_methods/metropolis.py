#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import numpy.random as nr
import theano
import scipy.linalg
import warnings
import logging

from ..distributions import draw_values
from .arraystep import ArrayStepShared, PopulationArrayStepShared, ArrayStep, metrop_select, Competence
from .compound import CompoundStep
import pymc3 as pm
from pymc3.theanof import floatX

import theano.tensor as tt

__all__ = ['Metropolis', 'DEMetropolis', 'DEMetropolisZ', 'BinaryMetropolis', 'BinaryGibbsMetropolis',
           'CategoricalGibbsMetropolis', 'UniformProposal', 'NormalProposal', 'CauchyProposal',
           'LaplaceProposal', 'PoissonProposal', 'MultivariateNormalProposal',
           'RecursiveDAProposal', 'MLDA']

# Available proposal distributions for Metropolis


class Proposal:
    def __init__(self, s):
        self.s = s


class NormalProposal(Proposal):
    def __call__(self):
        return nr.normal(scale=self.s)


class UniformProposal(Proposal):
    def __call__(self):
        return nr.uniform(low=-self.s, high=self.s, size=len(self.s))


class CauchyProposal(Proposal):
    def __call__(self):
        return nr.standard_cauchy(size=np.size(self.s)) * self.s


class LaplaceProposal(Proposal):
    def __call__(self):
        size = np.size(self.s)
        return (nr.standard_exponential(size=size) - nr.standard_exponential(size=size)) * self.s


class PoissonProposal(Proposal):
    def __call__(self):
        return nr.poisson(lam=self.s, size=np.size(self.s)) - self.s


class MultivariateNormalProposal(Proposal):
    def __init__(self, s):
        n, m = s.shape
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.n = n
        self.chol = scipy.linalg.cholesky(s, lower=True)

    def __call__(self, num_draws=None):
        if num_draws is not None:
            b = np.random.randn(self.n, num_draws)
            return np.dot(self.chol, b).T
        else:
            b = np.random.randn(self.n)
            return np.dot(self.chol, b)


class RecursiveDAProposal(Proposal):
    """
    Recursive Delayed Acceptance proposal to be used with MLDA step sampler.
    Recursively calls an MLDA sampler if level > 0 and calls Metropolis/DEMetropolisZ
    sampler if level = 0. The sampler generates subsampling_rate samples and
    the last one is used as a proposal. Results in a hierarchy of chains
    each of which is used to propose samples to the chain above.
    """
    def __init__(self, next_step_method, next_model,
                 tune, subsampling_rate):

        self.next_step_method = next_step_method
        self.next_model = next_model
        self.tune = tune
        self.subsampling_rate = subsampling_rate
        self.tuning_end_trigger = True
        self.trace = None

    def __call__(self, q0_dict):
        """Returns proposed sample  given the current sample
        in dictionary form (q0_dict)."""

        # Logging is reduced to avoid extensive console output
        # during multiple recursive calls of subsample()
        _log = logging.getLogger('pymc3')
        _log.setLevel(logging.ERROR)

        with self.next_model:
            # Check if the tuning flag has been set to False
            # in which case tuning is stopped. The flag is set
            # to False (by MLDA's astep) when the burn-in
            # iterations of the highest-level MLDA sampler run out.
            # The change propagates to all levels.

            if self.tune:
                # Subsample in tuning mode
                self.trace = pm.subsample(draws=0, step=self.next_step_method,
                                          start=q0_dict, trace=self.trace,
                                          tune=self.subsampling_rate)
            else:
                # Sample in normal mode without tuning
                # If DEMetropolisZ is the base sampler a flag is raised to
                # make sure that history is edited after tuning ends
                if self.tuning_end_trigger and isinstance(self.next_step_method, DEMetropolisZ):
                    self.next_step_method.tuning_end_trigger = True
                self.trace = pm.subsample(draws=self.subsampling_rate,
                                          step=self.next_step_method,
                                          start=q0_dict, trace=self.trace)
                self.tuning_end_trigger = False
                # If DEMetropolisZ is the base sampler the flag is set to False
                # to avoid further deletion of samples history
                if isinstance(self.next_step_method, DEMetropolisZ):
                    self.next_step_method.tuning_end_trigger = False

        # set logging back to normal
        _log.setLevel(logging.NOTSET)

        return self.trace.point(-1)


class Metropolis(ArrayStepShared):
    """
    Metropolis-Hastings sampling step

    Parameters
    ----------
    vars: list
        List of variables for sampler
    S: standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist: function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to normal.
    scaling: scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune: bool
        Flag for tuning. Defaults to True.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to Theano functions
    """
    name = 'metropolis'

    default_blocked = False
    generates_stats = True
    stats_dtypes = [{
        'accept': np.float64,
        'accepted': np.bool,
        'tune': np.bool,
        'scaling': np.float64,
    }]

    def __init__(self, vars=None, S=None, proposal_dist=None, scaling=1.,
                 tune=True, tune_interval=100, model=None, mode=None, **kwargs):

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(sum(v.dsize for v in vars))

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        elif S.ndim == 1:
            self.proposal_dist = NormalProposal(S)
        elif S.ndim == 2:
            self.proposal_dist = MultivariateNormalProposal(S)
        else:
            raise ValueError("Invalid rank for variance: %s" % S.ndim)

        self.scaling = np.atleast_1d(scaling).astype('d')
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        # Determine type of variables
        self.discrete = np.concatenate(
            [[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in vars])
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        # remember initial settings before tuning so they can be reset
        self._untuned_settings = dict(
            scaling=self.scaling,
            steps_until_tune=tune_interval,
            accepted=self.accepted
        )

        self.mode = mode

        # flag to indicate this stepper was instantiated within an MLDA stepper
        # if not, tuning parameters are reset when _iter_sample() is called
        self.is_mlda_base = kwargs.pop("is_mlda_base", False)

        # flag to indicate this stepper was instantiated within an MLDA stepper
        # and variance reduction is activated - forces Metropolis to store
        # quantities of interest in a register if True
        self.mlda_variance_reduction = kwargs.pop("mlda_variance_reduction", False)

        if self.mlda_variance_reduction:
            # Subsampling rate of MLDA sampler one level up
            self.mlda_subsampling_rate_above = kwargs.pop("mlda_subsampling_rate_above")
            self.sub_counter = 0
            self.Q_last = np.nan
            self.Q_reg = [np.nan] * self.mlda_subsampling_rate_above
            self.acceptance_reg = [None] * self.mlda_subsampling_rate_above
            self.model = model

        shared = pm.make_shared_replacements(vars, model)
        if self.mlda_variance_reduction:
            self.delta_logp = delta_logp_inverse(model.logpt, vars, shared)
        else:
            self.delta_logp = delta_logp(model.logpt, vars, shared)
        super().__init__(vars, shared)

    def reset_tuning(self):
        """Resets the tuned sampler parameters to their initial values.
           Skipped if stepper is a bottom-level stepper in MLDA."""
        if not self.is_mlda_base:
            for attr, initial_value in self._untuned_settings.items():
                setattr(self, attr, initial_value)
        return

    def astep(self, q0):
        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(
                self.scaling, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        delta = self.proposal_dist() * self.scaling

        if self.any_discrete:
            if self.all_discrete:
                delta = np.round(delta, 0).astype('int64')
                q0 = q0.astype('int64')
                q = (q0 + delta).astype('int64')
            else:
                delta[self.discrete] = np.round(
                    delta[self.discrete], 0)
                q = (q0 + delta)
        else:
            q = floatX(q0 + delta)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        if self.is_mlda_base and self.mlda_variance_reduction:
            if accepted:
                self.Q_last = self.model.Q.get_value()
            if self.sub_counter == self.mlda_subsampling_rate_above:
                self.sub_counter = 0
            self.Q_reg[self.sub_counter] = self.Q_last
            self.acceptance_reg[self.sub_counter] = accepted
            self.sub_counter += 1

        self.steps_until_tune -= 1

        stats = {
            'tune': self.tune,
            'scaling': self.scaling,
            'accept': np.exp(accept),
            'accepted': accepted,
        }

        return q_new, [stats]

    @staticmethod
    def competence(var, has_grad):
        return Competence.COMPATIBLE


def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """
    if acc_rate < 0.001:
        # reduce by 90 percent
        return scale * 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        return scale * 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        return scale * 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        return scale * 10.0
    elif acc_rate > 0.75:
        # increase by double
        return scale * 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        return scale * 1.1

    return scale


class BinaryMetropolis(ArrayStep):
    """Metropolis-Hastings optimized for binary variables

    Parameters
    ----------
    vars: list
        List of variables for sampler
    scaling: scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune: bool
        Flag for tuning. Defaults to True.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """
    name = 'binary_metropolis'

    generates_stats = True
    stats_dtypes = [{
        'accept': np.float64,
        'tune': np.bool,
        'p_jump': np.float64,
    }]

    def __init__(self, vars, scaling=1., tune=True, tune_interval=100, model=None):

        model = pm.modelcontext(model)

        self.scaling = scaling
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        if not all([v.dtype in pm.discrete_types for v in vars]):
            raise ValueError(
                'All variables must be Bernoulli for BinaryMetropolis')

        super().__init__(vars, [model.fastlogp])

    def astep(self, q0, logp):

        # Convert adaptive_scale_factor to a jump probability
        p_jump = 1. - .5 ** self.scaling

        rand_array = nr.random(q0.shape)
        q = np.copy(q0)
        # Locations where switches occur, according to p_jump
        switch_locs = (rand_array < p_jump)
        q[switch_locs] = True - q[switch_locs]

        accept = logp(q) - logp(q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        stats = {
            'tune': self.tune,
            'accept': np.exp(accept),
            'p_jump': p_jump,
        }

        return q_new, [stats]

    @staticmethod
    def competence(var):
        '''
        BinaryMetropolis is only suitable for binary (bool)
        and Categorical variables with k=1.
        '''
        distribution = getattr(
            var.distribution, 'parent_dist', var.distribution)
        if isinstance(distribution, pm.Bernoulli) or (var.dtype in pm.bool_types):
            return Competence.COMPATIBLE
        elif isinstance(distribution, pm.Categorical) and (distribution.k == 2):
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE


class BinaryGibbsMetropolis(ArrayStep):
    """A Metropolis-within-Gibbs step method optimized for binary variables

    Parameters
    ----------
    vars: list
        List of variables for sampler
    order: list or 'random'
        List of integers indicating the Gibbs update order
        e.g., [0, 2, 1, ...]. Default is random
    transit_p: float
        The diagonal of the transition kernel. A value > .5 gives anticorrelated proposals,
        which resulting in more efficient antithetical sampling.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """
    name = 'binary_gibbs_metropolis'

    def __init__(self, vars, order='random', transit_p=.8, model=None):

        model = pm.modelcontext(model)

        # transition probabilities
        self.transit_p = transit_p

        self.dim = sum(v.dsize for v in vars)

        if order == 'random':
            self.shuffle_dims = True
            self.order = list(range(self.dim))
        else:
            if sorted(order) != list(range(self.dim)):
                raise ValueError('Argument \'order\' has to be a permutation')
            self.shuffle_dims = False
            self.order = order

        if not all([v.dtype in pm.discrete_types for v in vars]):
            raise ValueError(
                'All variables must be binary for BinaryGibbsMetropolis')

        super().__init__(vars, [model.fastlogp])

    def astep(self, q0, logp):
        order = self.order
        if self.shuffle_dims:
            nr.shuffle(order)

        q = np.copy(q0)
        logp_curr = logp(q)

        for idx in order:
            # No need to do metropolis update if the same value is proposed,
            # as you will get the same value regardless of accepted or reject
            if nr.rand() < self.transit_p:
                curr_val, q[idx] = q[idx], True - q[idx]
                logp_prop = logp(q)
                q[idx], accepted = metrop_select(logp_prop - logp_curr, q[idx], curr_val)
                if accepted:
                    logp_curr = logp_prop

        return q

    @staticmethod
    def competence(var):
        '''
        BinaryMetropolis is only suitable for Bernoulli
        and Categorical variables with k=2.
        '''
        distribution = getattr(
            var.distribution, 'parent_dist', var.distribution)
        if isinstance(distribution, pm.Bernoulli) or (var.dtype in pm.bool_types):
            return Competence.IDEAL
        elif isinstance(distribution, pm.Categorical) and (distribution.k == 2):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


class CategoricalGibbsMetropolis(ArrayStep):
    """A Metropolis-within-Gibbs step method optimized for categorical variables.
       This step method works for Bernoulli variables as well, but it is not
       optimized for them, like BinaryGibbsMetropolis is. Step method supports
       two types of proposals: A uniform proposal and a proportional proposal,
       which was introduced by Liu in his 1996 technical report
       "Metropolized Gibbs Sampler: An Improvement".
    """
    name = 'categorical_gibbs_metropolis'

    def __init__(self, vars, proposal='uniform', order='random', model=None):

        model = pm.modelcontext(model)
        vars = pm.inputvars(vars)

        dimcats = []
        # The above variable is a list of pairs (aggregate dimension, number
        # of categories). For example, if vars = [x, y] with x being a 2-D
        # variable with M categories and y being a 3-D variable with N
        # categories, we will have dimcats = [(0, M), (1, M), (2, N), (3, N), (4, N)].
        for v in vars:
            distr = getattr(v.distribution, 'parent_dist', v.distribution)
            if isinstance(distr, pm.Categorical):
                k = draw_values([distr.k])[0]
            elif isinstance(distr, pm.Bernoulli) or (v.dtype in pm.bool_types):
                k = 2
            else:
                raise ValueError('All variables must be categorical or binary' +
                                 'for CategoricalGibbsMetropolis')
            start = len(dimcats)
            dimcats += [(dim, k) for dim in range(start, start + v.dsize)]

        if order == 'random':
            self.shuffle_dims = True
            self.dimcats = dimcats
        else:
            if sorted(order) != list(range(len(dimcats))):
                raise ValueError('Argument \'order\' has to be a permutation')
            self.shuffle_dims = False
            self.dimcats = [dimcats[j] for j in order]

        if proposal == 'uniform':
            self.astep = self.astep_unif
        elif proposal == 'proportional':
            # Use the optimized "Metropolized Gibbs Sampler" described in Liu96.
            self.astep = self.astep_prop
        else:
            raise ValueError('Argument \'proposal\' should either be ' +
                    '\'uniform\' or \'proportional\'')

        super().__init__(vars, [model.fastlogp])

    def astep_unif(self, q0, logp):
        dimcats = self.dimcats
        if self.shuffle_dims:
            nr.shuffle(dimcats)

        q = np.copy(q0)
        logp_curr = logp(q)

        for dim, k in dimcats:
            curr_val, q[dim] = q[dim], sample_except(k, q[dim])
            logp_prop = logp(q)
            q[dim], accepted = metrop_select(logp_prop - logp_curr, q[dim], curr_val)
            if accepted:
                logp_curr = logp_prop
        return q

    def astep_prop(self, q0, logp):
        dimcats = self.dimcats
        if self.shuffle_dims:
            nr.shuffle(dimcats)

        q = np.copy(q0)
        logp_curr = logp(q)

        for dim, k in dimcats:
            logp_curr = self.metropolis_proportional(q, logp, logp_curr, dim, k)

        return q

    def metropolis_proportional(self, q, logp, logp_curr, dim, k):
        given_cat = int(q[dim])
        log_probs = np.zeros(k)
        log_probs[given_cat] = logp_curr
        candidates = list(range(k))
        for candidate_cat in candidates:
            if candidate_cat != given_cat:
                q[dim] = candidate_cat
                log_probs[candidate_cat] = logp(q)
        probs = softmax(log_probs)
        prob_curr, probs[given_cat] = probs[given_cat], 0.0
        probs /= (1.0 - prob_curr)
        proposed_cat = nr.choice(candidates, p = probs)
        accept_ratio = (1.0 - prob_curr) / (1.0 - probs[proposed_cat])
        if not np.isfinite(accept_ratio) or nr.uniform() >= accept_ratio:
            q[dim] = given_cat
            return logp_curr
        q[dim] = proposed_cat
        return log_probs[proposed_cat]

    @staticmethod
    def competence(var):
        '''
        CategoricalGibbsMetropolis is only suitable for Bernoulli and
        Categorical variables.
        '''
        distribution = getattr(
            var.distribution, 'parent_dist', var.distribution)
        if isinstance(distribution, pm.Categorical):
            if distribution.k > 2:
                return Competence.IDEAL
            return Competence.COMPATIBLE
        elif isinstance(distribution, pm.Bernoulli) or (var.dtype in pm.bool_types):
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE


class DEMetropolis(PopulationArrayStepShared):
    """
    Differential Evolution Metropolis sampling step.

    Parameters
    ----------
    lamb: float
        Lambda parameter of the DE proposal mechanism. Defaults to 2.38 / sqrt(2 * ndim)
    vars: list
        List of variables for sampler
    S: standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist: function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to Uniform(-S,+S).
    scaling: scalar or array
        Initial scale factor for epsilon. Defaults to 0.001
    tune: str
        Which hyperparameter to tune. Defaults to None, but can also be 'scaling' or 'lambda'.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to Theano functions

    References
    ----------
    .. [Braak2006] Cajo C.F. ter Braak (2006).
        A Markov Chain Monte Carlo version of the genetic algorithm
        Differential Evolution: easy Bayesian computing for real parameter spaces.
        Statistics and Computing
        `link <https://doi.org/10.1007/s11222-006-8769-1>`__
    """
    name = 'DEMetropolis'

    default_blocked = True
    generates_stats = True
    stats_dtypes = [{
        'accept': np.float64,
        'accepted': np.bool,
        'tune': np.bool,
        'scaling': np.float64,
        'lambda': np.float64,
    }]

    def __init__(self, vars=None, S=None, proposal_dist=None, lamb=None, scaling=0.001,
                 tune=None, tune_interval=100, model=None, mode=None, **kwargs):

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(model.ndim)

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        else:
            self.proposal_dist = UniformProposal(S)

        self.scaling = np.atleast_1d(scaling).astype('d')
        if lamb is None:
            # default to the optimal lambda for normally distributed targets
            lamb = 2.38 / np.sqrt(2 * model.ndim)
        self.lamb = float(lamb)
        if tune not in {None, 'scaling', 'lambda'}:
            raise ValueError('The parameter "tune" must be one of {None, scaling, lambda}')
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        self.mode = mode

        shared = pm.make_shared_replacements(vars, model)
        self.delta_logp = delta_logp(model.logpt, vars, shared)
        super().__init__(vars, shared)

    def astep(self, q0):
        if not self.steps_until_tune and self.tune:
            if self.tune == 'scaling':
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
            elif self.tune == 'lambda':
                self.lamb = tune(self.lamb, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        epsilon = self.proposal_dist() * self.scaling

        # differential evolution proposal
        # select two other chains
        ir1, ir2 = np.random.choice(self.other_chains, 2, replace=False)
        r1 = self.bij.map(self.population[ir1])
        r2 = self.bij.map(self.population[ir2])
        # propose a jump
        q = floatX(q0 + self.lamb * (r1 - r2) + epsilon)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        self.steps_until_tune -= 1

        stats = {
            'tune': self.tune,
            'scaling': self.scaling,
            'lambda': self.lamb,
            'accept': np.exp(accept),
            'accepted': accepted
        }

        return q_new, [stats]

    @staticmethod
    def competence(var, has_grad):
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE



class DEMetropolisZ(ArrayStepShared):
    """
    Adaptive Differential Evolution Metropolis sampling step that uses the past to inform jumps.

    Parameters
    ----------
    lamb: float
        Lambda parameter of the DE proposal mechanism. Defaults to 2.38 / sqrt(2 * ndim)
    vars: list
        List of variables for sampler
    S: standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist: function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to Uniform(-S,+S).
    scaling: scalar or array
        Initial scale factor for epsilon. Defaults to 0.001
    tune: str
        Which hyperparameter to tune. Defaults to 'lambda', but can also be 'scaling' or None.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    tune_drop_fraction: float
        Fraction of tuning steps that will be removed from the samplers history when the tuning ends.
        Defaults to 0.9 - keeping the last 10% of tuning steps for good mixing while removing 90% of
        potentially unconverged tuning positions.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to Theano functions

    References
    ----------
    .. [Braak2006] Cajo C.F. ter Braak (2006).
        Differential Evolution Markov Chain with snooker updater and fewer chains.
        Statistics and Computing
        `link <https://doi.org/10.1007/s11222-008-9104-9>`__
    """
    name = 'DEMetropolisZ'

    default_blocked = True
    generates_stats = True
    stats_dtypes = [{
        'accept': np.float64,
        'accepted': np.bool,
        'tune': np.bool,
        'scaling': np.float64,
        'lambda': np.float64,
    }]

    def __init__(self, vars=None, S=None, proposal_dist=None, lamb=None, scaling=0.001,
                 tune='lambda', tune_interval=100, tune_drop_fraction:float=0.9, model=None, mode=None, **kwargs):
        model = pm.modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(model.ndim)

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        else:
            self.proposal_dist = UniformProposal(S)

        self.scaling = np.atleast_1d(scaling).astype('d')
        if lamb is None:
            # default to the optimal lambda for normally distributed targets
            lamb = 2.38 / np.sqrt(2 * model.ndim)
        self.lamb = float(lamb)
        if tune not in {None, 'scaling', 'lambda'}:
            raise ValueError('The parameter "tune" must be one of {None, scaling, lambda}')
        self.tune = True
        self.tune_target = tune
        self.tune_interval = tune_interval
        self.tune_drop_fraction = tune_drop_fraction
        self.steps_until_tune = tune_interval
        self.accepted = 0

        # cache local history for the Z-proposals
        self._history = []
        # remember initial settings before tuning so they can be reset
        self._untuned_settings = dict(
            scaling=self.scaling,
            lamb=self.lamb,
            steps_until_tune=tune_interval,
            accepted=self.accepted
        )

        self.mode = mode

        # flag to indicate this stepper was instantiated within an MLDA stepper
        # if not, tuning parameters are reset when _iter_sample() is called
        self.is_mlda_base = kwargs.pop("is_mlda_base", False)
        # flag used for signifying the end of tuning when is_mlda_base is True
        self.tuning_end_trigger = False

        # flag to indicate this stepper was instantiated within an MLDA stepper
        # and variance reduction is activated - forces DEMetropolisZ to temparirily store
        # quantities of interest in a register if True
        self.mlda_variance_reduction = kwargs.pop("mlda_variance_reduction", False)

        if self.mlda_variance_reduction:
            # Subsampling rate of MLDA sampler one level up
            self.mlda_subsampling_rate_above = kwargs.pop("mlda_subsampling_rate_above")
            self.sub_counter = 0
            self.Q_last = np.nan
            self.Q_reg = [np.nan] * self.mlda_subsampling_rate_above
            self.acceptance_reg = [None] * self.mlda_subsampling_rate_above
            self.model = model

        shared = pm.make_shared_replacements(vars, model)
        if self.mlda_variance_reduction:
            self.delta_logp = delta_logp_inverse(model.logpt, vars, shared)
        else:
            self.delta_logp = delta_logp(model.logpt, vars, shared)

        super().__init__(vars, shared)

    def reset_tuning(self):
        """Resets the tuned sampler parameters and history to their initial values.
        Skipped if stepper is a bottom-level stepper in MLDA."""
        if not self.is_mlda_base:
            # history can't be reset via the _untuned_settings dict because it's a list
            self._history = []
            for attr, initial_value in self._untuned_settings.items():
                setattr(self, attr, initial_value)
        return

    def astep(self, q0):
        # same tuning scheme as DEMetropolis
        if not self.steps_until_tune and self.tune:
            if self.tune_target == 'scaling':
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
            elif self.tune_target == 'lambda':
                self.lamb = tune(self.lamb, self.accepted / float(self.tune_interval))

            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        epsilon = self.proposal_dist() * self.scaling

        it = len(self._history)
        # use the DE-MCMC-Z proposal scheme as soon as the history has 2 entries
        if it > 1:
            # differential evolution proposal
            # select two other chains
            iz1 = np.random.randint(it)
            iz2 = np.random.randint(it)
            while iz2 == iz1:
                iz2 = np.random.randint(it)

            z1 = self._history[iz1]
            z2 = self._history[iz2]
            # propose a jump
            q = floatX(q0 + self.lamb * (z1 - z2) + epsilon)
        else:
            # propose just with noise in the first 2 iterations
            q = floatX(q0 + epsilon)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted
        self._history.append(q_new)

        if self.is_mlda_base and self.mlda_variance_reduction:
            if accepted:
                self.Q_last = self.model.Q.get_value()
            if self.sub_counter == self.mlda_subsampling_rate_above:
                self.sub_counter = 0
            self.Q_reg[self.sub_counter] = self.Q_last
            self.acceptance_reg[self.sub_counter] = accepted
            self.sub_counter += 1

        self.steps_until_tune -= 1

        stats = {
            'tune': self.tune,
            'scaling': self.scaling,
            'lambda': self.lamb,
            'accept': np.exp(accept),
            'accepted': accepted
        }

        return q_new, [stats]

    def stop_tuning(self):
        """At the end of the tuning phase, this method removes the first x% of the history
        so future proposals are not informed by unconverged tuning iterations.
        Does not run when used as part of MLDA, except for one time after the end
        of tuning.
        """
        if not self.is_mlda_base or (self.is_mlda_base and self.tuning_end_trigger):
            it = len(self._history)
            n_drop = int(self.tune_drop_fraction * it)
            self._history = self._history[n_drop:]
        return super().stop_tuning()

    @staticmethod
    def competence(var, has_grad):
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE


class MLDA(ArrayStepShared):
    """
    Multi-Level Delayed Acceptance (MLDA) sampling step that uses coarse
    approximations of a fine model to construct proposals in multiple levels.

    MLDA creates a hierarchy of MCMC chains. Chains sample from different
    posteriors that ideally should be approximations of the fine (top-level)
    posterior and require less computational effort to evaluate their likelihood.

    Each chain runs for a fixed number of iterations (subsampling_rate) and then
    the last sample generated is used as a proposal for the chain in the level
    above. The bottom-level chain is a Metropolis or DEMetropolisZ sampler.
    The algorithm achieves higher acceptance rate and effective sample sizes
    than other samplers if the coarse models are sufficiently good approximations
    of the fine one.

    Parameters
    ----------
    coarse_models : list
        List of coarse (multi-level) models, where the first model
        is the coarsest one (level=0) and the last model is the
        second finest one (level=L-1 where L is the number of levels).
        Note this list excludes the model passed to the model
        argument above, which is the finest available.
    vars : list
        List of variables for sampler
    base_sampler : string
        Sampler used in the base (coarsest) chain. Can be 'Metropolis' or
        'DEMetropolisZ'. Defaults to 'DEMetropolisZ'.
    base_S : standard deviation of base proposal covariance matrix
        Some measure of variance to parameterize base proposal distribution
    base_proposal_dist : function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to normal. This is the proposal used in the
        coarsest (base) chain, i.e. level=0.
    base_scaling : scalar or array
        Initial scale factor for base proposal. Defaults to 1. if base_sampler
        is 'Metropolis' and to 0.001 if base_sampler is 'DEMetropolisZ'.
    tune : bool
        Flag for tuning in the base proposal. If base_sampler is 'Metropolis' it
        should be True or False and defaults to True. Note that
        this is overidden by the tune parameter in sample(). For example when calling
        step=MLDA(tune=False, ...) and then sample(step=step, tune=200, ...),
        tuning will be activated for the first 200 steps. If base_sampler is
        'DEMetropolisZ', it should be True. For 'DEMetropolisZ', there is a separate
        argument base_tune_target which allows modifying the type of tuning.
    base_tune_target: string
        Defines the type of tuning that is performed when base_sampler is
        'DEMetropolisZ'. Allowable values are 'lambda, 'scaling' or None and
        it defaults to 'lambda'.
    base_tune_interval : int
        The frequency of tuning for the base proposal. Defaults to 100
        iterations.
    base_lamb : float
        Lambda parameter of the base level DE proposal mechanism. Only applicable when
        base_sampler is 'DEMetropolisZ'. Defaults to 2.38 / sqrt(2 * ndim)
    base_tune_drop_fraction: float
        Fraction of tuning steps that will be removed from the base level samplers
        history when the tuning ends. Only applicable when base_sampler is
        'DEMetropolisZ'. Defaults to 0.9 - keeping the last 10% of tuning steps
        for good mixing while removing 90% of potentially unconverged tuning positions.
    model : PyMC Model
        Optional model for sampling step. Defaults to None
        (taken from context). This model should be the finest of all
        multilevel models.
    mode :  string or `Mode` instance.
        Compilation mode passed to Theano functions
    subsampling_rates : integer or list of integers
        One interger for all levels or a list with one number for each level
        (excluding the finest level).
        This is the number of samples generated in level l-1 to propose a sample
        for level l for all l levels (excluding the finest level). The length of
        the list needs to be the same as the length of coarse_models.
    base_blocked : bool
        To flag to choose whether base sampler (level=0) is a
        Compound Metropolis step (base_blocked=False)
        or a blocked Metropolis step (base_blocked=True).
    variance_reduction: bool
        Calculate and store quantities of interest and quantity of interest
        differences between levels to enable computing a variance-reduced
        sum of the quantity of interest after sampling.
    store_Q_fine: bool
        Store the values of the quantity of interest from the fine chain.

    Examples
    ----------
    .. code:: ipython

        >>> import pymc3 as pm
        ... datum = 1
        ...
        ... with pm.Model() as coarse_model:
        ...     x = pm.Normal("x", mu=0, sigma=10)
        ...     y = pm.Normal("y", mu=x, sigma=1, observed=datum - 0.1)
        ...
        ... with pm.Model():
        ...     x = pm.Normal("x", mu=0, sigma=10)
        ...     y = pm.Normal("y", mu=x, sigma=1, observed=datum)
        ...     step_method = pm.MLDA(coarse_models=[coarse_model]
        ...                           subsampling_rates=5)
        ...     trace = pm.sample(draws=500, chains=2,
        ...                       tune=100, step=step_method,
        ...                       random_seed=123)
        ...
        ... pm.summary(trace)
            mean     sd	     hpd_3%	   hpd_97%
        x	0.982	1.026	 -0.994	   2.902

    A more complete example of how to use MLDA in a realistic
    multilevel problem can be found in:
    pymc3/docs/source/notebooks/multi-level_groundwater_flow_with_MLDA.ipynb

    References
    ----------
    .. [Dodwell2019] Dodwell, Tim & Ketelsen, Chris & Scheichl,
    Robert & Teckentrup, Aretha. (2019).
    Multilevel Markov Chain Monte Carlo.
    SIAM Review. 61. 509-545.
        `link <https://doi.org/10.1137/19M126966X>`__
    """
    name = 'mlda'

    # All levels use block sampling,
    # except level 0 where the user can choose
    default_blocked = True
    generates_stats = True

    # These stats are extended within __init__
    stats_dtypes = [{
        'accept': np.float64,
        'accepted': np.bool,
        'tune': np.bool,
        'base_scaling': object
    }]

    def __init__(self, coarse_models, vars=None, base_sampler='DEMetropolisZ',
                 base_S=None, base_proposal_dist=None, base_scaling=None,
                 tune=True, base_tune_target='lambda', base_tune_interval=100,
                 base_lamb=None, base_tune_drop_fraction=0.9, model=None, mode=None,
                 subsampling_rates=5, base_blocked=False, variance_reduction=False,
                 store_Q_fine=False, **kwargs):

        # this variable is used to identify MLDA objects which are
        # not in the finest level (i.e. child MLDA objects)
        self.is_child = kwargs.get("is_child", False)
        if not self.is_child:
            warnings.warn(
                'The MLDA implementation in PyMC3 is very young. '
                'You should be extra critical about its results.'
            )

        model = pm.modelcontext(model)

        # assign internal state
        self.coarse_models = coarse_models
        if not isinstance(coarse_models, list):
            raise ValueError("MLDA step method cannot use "
                             "coarse_models if it is not a list")
        if len(self.coarse_models) == 0:
            raise ValueError("MLDA step method was given an empty "
                             "list of coarse models. Give at least "
                             "one coarse model.")
        self.model = model
        self.variance_reduction = variance_reduction
        self.store_Q_fine = store_Q_fine

        # check that certain requirements hold
        # for the variance reduction feature to work
        if self.variance_reduction or self.store_Q_fine:
            if not hasattr(self.model, 'Q'):
                raise AttributeError("Model given to MLDA does not contain"
                                     "variable 'Q'. You need to include"
                                     "the variable in the model definition"
                                     "for variance reduction to work or"
                                     "for storing the fine Q."
                                     "Use pm.Data() to define it.")
            if not isinstance(self.model.Q, tt.sharedvar.TensorSharedVariable):
                raise TypeError("The variable 'Q' in the model definition is not of type "
                                "'TensorSharedVariable'. Use pm.Data() to define the"
                                "variable.")

        if isinstance(subsampling_rates, int):
            self.subsampling_rates = [subsampling_rates] * len(self.coarse_models)
        else:
            if len(subsampling_rates) != len(self.coarse_models):
                raise ValueError(f"List of subsampling rates needs to have the same "
                                 f"length as list of coarse models but the lengths "
                                 f"were {len(subsampling_rates)}, {len(self.coarse_models)}")
            self.subsampling_rates = subsampling_rates

        if self.is_child:
            # this is the subsampling rate applied to the current level
            # it is stored in the level above and transferred here
            self.subsampling_rate_above = kwargs.get("subsampling_rate_above", None)
        self.num_levels = len(self.coarse_models) + 1
        self.base_sampler = base_sampler

        # VR is not compatible with compound base samplers so an automatic conversion
        # to a block sampler happens here if
        if self.variance_reduction and self.base_sampler == 'Metropolis' and not base_blocked:
            warnings.warn(
                'Variance reduction is not compatible with non-blocked (compound) samplers.'
                'Automatically switching to a blocked Metropolis sampler.'
            )
            self.base_blocked = True
        else:
            self.base_blocked = base_blocked
        self.next_model = self.coarse_models[-1]
        self.base_S = base_S
        self.base_proposal_dist = base_proposal_dist

        if base_scaling is None:
            if self.base_sampler == 'Metropolis':
                self.base_scaling = 1.
            else:
                self.base_scaling = 0.001
        else:
            self.base_scaling = float(base_scaling)

        self.tune = tune
        if not self.tune and self.base_sampler == 'DEMetropolisZ':
            raise ValueError(f"The argument tune was set to False while using"
                             f" a 'DEMetropolisZ' base sampler. 'DEMetropolisZ' "
                             f" tune needs to be True.")

        self.base_tune_target = base_tune_target
        self.base_tune_interval = base_tune_interval
        self.base_lamb = base_lamb
        self.base_tune_drop_fraction = float(base_tune_drop_fraction)
        self.mode = mode
        self.base_scaling_stats = None
        if self.base_sampler == 'DEMetropolisZ':
            self.base_lambda_stats = None

        # Process model variables
        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)
        self.vars = vars
        self.var_names = [var.name for var in self.vars]

        self.accepted = 0

        # Construct theano function for current-level model likelihood
        # (for use in acceptance)
        shared = pm.make_shared_replacements(vars,
                                             model)
        self.delta_logp = delta_logp_inverse(model.logpt,
                                             vars,
                                             shared)

        # Construct theano function for next-level model likelihood
        # (for use in acceptance)
        next_model = pm.modelcontext(self.next_model)
        vars_next = [var for var in next_model.vars if var.name in self.var_names]
        vars_next = pm.inputvars(vars_next)
        shared_next = pm.make_shared_replacements(vars_next,
                                                  next_model)
        self.delta_logp_next = delta_logp(next_model.logpt,
                                          vars_next,
                                          shared_next)

        super().__init__(vars, shared)

        # initialise complete step method hierarchy
        if self.num_levels == 2:
            with self.next_model:
                # make sure the correct variables are selected from next_model
                vars_next = [var for var in self.next_model.vars
                             if var.name in self.var_names]

                # create kwargs
                if self.variance_reduction:
                    base_kwargs = {"is_mlda_base": True,
                                   "mlda_subsampling_rate_above": self.subsampling_rates[-1],
                                   "mlda_variance_reduction": True}
                else:
                    base_kwargs = {"is_mlda_base": True}

                if self.base_sampler == 'Metropolis':
                    # Metropolis sampler in base level (level=0), targeting self.next_model
                    # The flag is_mlda_base is set to True to prevent tuning reset
                    # between MLDA iterations - note that Metropolis is used
                    # with only one chain and therefore the scaling reset issue
                    # (see issue #3733 in GitHub) will not appear here
                    self.next_step_method = pm.Metropolis(vars=vars_next,
                                                          proposal_dist=self.base_proposal_dist,
                                                          S=self.base_S,
                                                          scaling=self.base_scaling, tune=self.tune,
                                                          tune_interval=self.base_tune_interval,
                                                          model=None,
                                                          mode=self.mode,
                                                          blocked=self.base_blocked,
                                                          ** base_kwargs)
                else:
                    # DEMetropolisZ sampler in base level (level=0), targeting self.next_model
                    self.next_step_method = pm.DEMetropolisZ(vars=vars_next,
                                                             S=self.base_S,
                                                             proposal_dist=self.base_proposal_dist,
                                                             lamb=self.base_lamb,
                                                             scaling=self.base_scaling,
                                                             tune=self.base_tune_target,
                                                             tune_interval=self.base_tune_interval,
                                                             tune_drop_fraction=self.base_tune_drop_fraction,
                                                             model=None,
                                                             mode=self.mode,
                                                             ** base_kwargs)
        else:
            # drop the last coarse model
            next_coarse_models = self.coarse_models[:-1]
            next_subsampling_rates = self.subsampling_rates[:-1]
            with self.next_model:
                # make sure the correct variables are selected from next_model
                vars_next = [var for var in self.next_model.vars
                             if var.name in self.var_names]

                # create kwargs
                if self.variance_reduction:
                    mlda_kwargs = {"is_child": True,
                                   "subsampling_rate_above": self.subsampling_rates[-1]}
                else:
                    mlda_kwargs = {"is_child": True}

                # MLDA sampler in some intermediate level, targeting self.next_model
                self.next_step_method = pm.MLDA(vars=vars_next, base_S=self.base_S,
                                                base_sampler=self.base_sampler,
                                                base_proposal_dist=self.base_proposal_dist,
                                                base_scaling=self.base_scaling,
                                                tune=self.tune,
                                                base_tune_target=self.base_tune_target,
                                                base_tune_interval=self.base_tune_interval,
                                                base_lamb=self.base_lamb,
                                                base_tune_drop_fraction=self.base_tune_drop_fraction,
                                                model=None, mode=self.mode,
                                                subsampling_rates=next_subsampling_rates,
                                                coarse_models=next_coarse_models,
                                                base_blocked=self.base_blocked,
                                                variance_reduction=self.variance_reduction,
                                                store_Q_fine=False,
                                                **mlda_kwargs)

        # instantiate the recursive DA proposal.
        # this is the main proposal used for
        # all levels (Recursive Delayed Acceptance)
        # (except for level 0 where the step method is
        # Metropolis/DEMetropolisZ and not MLDA)
        self.proposal_dist = RecursiveDAProposal(self.next_step_method,
                                                 self.next_model,
                                                 self.tune,
                                                 self.subsampling_rates[-1])

        # add 'base_lambda' to stats if 'DEMetropolisZ' is used
        if self.base_sampler == 'DEMetropolisZ':
            self.stats_dtypes[0]['base_lambda'] = np.float64

        # initialise necessary variables for doing variance reduction
        if self.variance_reduction:
            self.sub_counter = 0
            self.Q_diff = []
            if self.is_child:
                self.Q_reg = [np.nan] * self.subsampling_rate_above
            if self.num_levels == 2:
                self.Q_base_full = []
            if not self.is_child:
                for level in range(self.num_levels - 1, 0, -1):
                    self.stats_dtypes[0][f'Q_{level}_{level - 1}'] = object
                self.stats_dtypes[0]['Q_0'] = object

        # initialise necessary variables for doing variance reduction or storing fine Q
        if self.variance_reduction or self.store_Q_fine:
            self.Q_last = np.nan
            self.Q_diff_last = np.nan
        if self.store_Q_fine and not self.is_child:
            self.stats_dtypes[0][f'Q_{self.num_levels - 1}'] = object

    def astep(self, q0):
        """One MLDA step, given current sample q0"""
        # Check if the tuning flag has been changed and, if yes,
        # change the proposal's tuning flag and reset self.accepted
        # This is triggered by _iter_sample while the highest-level MLDA step
        # method is running. It then propagates to all levels.
        if self.proposal_dist.tune != self.tune:
            self.proposal_dist.tune = self.tune
            # set tune in sub-methods of compound stepper explicitly because
            # it is not set within sample.py (only the CompoundStep's tune flag is)
            if isinstance(self.next_step_method, CompoundStep):
                for method in self.next_step_method.methods:
                    method.tune = self.tune
            self.accepted = 0

        # Convert current sample from numpy array ->
        # dict before feeding to proposal
        q0_dict = self.bij.rmap(q0)

        # Call the recursive DA proposal to get proposed sample
        # and convert dict -> numpy array
        q = self.bij.map(self.proposal_dist(q0_dict))

        # Evaluate MLDA acceptance log-ratio
        # If proposed sample from lower level is the same as current one,
        # do not calculate likelihood, just set accept to 0.0
        if (q == q0).all():
            accept = np.float(0.0)
            skipped_logp = True
        else:
            accept = self.delta_logp(q, q0) + self.delta_logp_next(q0, q)
            skipped_logp = False

        # Accept/reject sample - next sample is stored in q_new
        q_new, accepted = metrop_select(accept, q, q0)
        if skipped_logp:
            accepted = False

        # Variance reduction
        self.update_vr_variables(accepted, skipped_logp)

        # Update acceptance counter
        self.accepted += accepted

        stats = {
            'tune': self.tune,
            'accept': np.exp(accept),
            'accepted': accepted
        }

        # Capture latest base chain scaling stats from next step method
        self.base_scaling_stats = {}
        if self.base_sampler == "DEMetropolisZ":
            self.base_lambda_stats = {}
        if isinstance(self.next_step_method, CompoundStep):
            # next method is Compound Metropolis
            scaling_list = []
            for method in self.next_step_method.methods:
                scaling_list.append(method.scaling)
            self.base_scaling_stats = {"base_scaling": np.array(scaling_list)}
        elif not isinstance(self.next_step_method, MLDA):
            # next method is any block sampler
            self.base_scaling_stats = {"base_scaling": np.array(self.next_step_method.scaling)}
            if self.base_sampler == "DEMetropolisZ":
                self.base_lambda_stats = {"base_lambda": self.next_step_method.lamb}
        else:
            # next method is MLDA - propagate dict from lower levels
            self.base_scaling_stats = self.next_step_method.base_scaling_stats
            if self.base_sampler == "DEMetropolisZ":
                self.base_lambda_stats = self.next_step_method.base_lambda_stats
        stats = {**stats, **self.base_scaling_stats}
        if self.base_sampler == "DEMetropolisZ":
            stats = {**stats, **self.base_lambda_stats}

        # Save the VR statistics to the stats dictionary (only happens in the
        # top MLDA level)
        if (self.variance_reduction or self.store_Q_fine) and not self.is_child:
            q_stats = {}
            if self.variance_reduction:
                m = self
                for level in range(self.num_levels - 1, 0, -1):
                    # save the Q differences for this level and iteration
                    q_stats[f'Q_{level}_{level - 1}'] = np.array(m.Q_diff)
                    # this makes sure Q_diff is reset for
                    # the next iteration
                    m.Q_diff = []
                    if level == 1:
                        break
                    m = m.next_step_method
                q_stats['Q_0'] = np.array(m.Q_base_full)
                m.Q_base_full = []
            if self.store_Q_fine:
                q_stats['Q_' + str(self.num_levels - 1)] = np.array(self.Q_last)
            stats = {**stats, **q_stats}

        return q_new, [stats]

    def update_vr_variables(self, accepted, skipped_logp):
        """Updates all the variables necessary for VR to work.

        Each level has a Q_last and Q_diff_last register which store
        the Q of the last accepted MCMC sample and the difference
        between the Q of the last accepted sample in this level and
        the Q of the last sample in the level below.

        These registers are updated here so that they can be exported later."""

        # if sample is accepted, update self.Q_last with the sample's Q value
        # runs only for VR or when store_Q_fine is True
        if self.variance_reduction or self.store_Q_fine:
            if accepted and not skipped_logp:
                self.Q_last = self.model.Q.get_value()

        if self.variance_reduction:
            # if this MLDA is not at the finest level, store Q_last in a
            # register Q_reg and increase sub_counter (until you reach
            # the subsampling rate, at which point you make it zero)
            # Q_reg will later be used by the level above to calculate differences
            if self.is_child:
                if self.sub_counter == self.subsampling_rate_above:
                    self.sub_counter = 0
                self.Q_reg[self.sub_counter] = self.Q_last
                self.sub_counter += 1

            # if MLDA is in the level above the base level, extract the
            # latest set of Q values from Q_reg in the base level
            # and add them to Q_base_full (which stores all the history of
            # Q values from the base level)
            if self.num_levels == 2:
                self.Q_base_full.extend(self.next_step_method.Q_reg)

            # if the sample is accepted, update Q_diff_last with the latest
            # difference between the Q of this level and the last Q of the
            # level below. If sample is not accepted, just keep the latest
            # accepted Q_diff
            if accepted and not skipped_logp:
                self.Q_diff_last = self.Q_last - self.next_step_method.Q_reg[self.subsampling_rates[-1] - 1]
            # Add the last accepted Q_diff to the list
            self.Q_diff.append(self.Q_diff_last)

    @staticmethod
    def competence(var, has_grad):
        """Return MLDA competence for given var/has_grad. MLDA currently works
        only with continuous variables."""
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE


def sample_except(limit, excluded):
    candidate = nr.choice(limit - 1)
    if candidate >= excluded:
        candidate += 1
    return candidate


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis = 0)


def delta_logp(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type('inarray1')

    logp1 = pm.CallableTensor(logp0)(inarray1)

    f = theano.function([inarray1, inarray0], logp1 - logp0)
    f.trust_input = True
    return f


def delta_logp_inverse(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type('inarray1')

    logp1 = pm.CallableTensor(logp0)(inarray1)

    f = theano.function([inarray1, inarray0], - logp0 + logp1)
    f.trust_input = True
    return f

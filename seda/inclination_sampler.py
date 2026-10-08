# Code based on Fields+2025: https://github.com/mjfields/cosi


import numpy as np
from packaging import version
import warnings
import emcee


class Probability(object):
    
    """
    The log-probability functions used by the MCMC estimation
    
    Parameters
    ----------
    vsini : float
        Line-of-sight stellar rotation velocity in km/s.
    e_vsini : float
        Uncertainty on vsini in km/s.
    rstar : float
        Stellar radius in solar radii.
    e_rstar : float
        Uncertainty on rstar in solar radii.
    prot : float
        Stellar rotation period in hours.
    e_prot : float
        Uncertainty on prot in hours.
    upperlimitvsini : bool, optional
        Is vsini only available as an upper limit? The default is False,
        i.e., vsini is measured with some confidence.
    
    """
    
    def __init__(self, vsini, e_vsini, rstar, e_rstar, prot, e_prot, upperlimitvsini=False):
        
        self.vsini = vsini # km/s
        self.e_vsini = e_vsini
        
        self.rstar = rstar # R_Sun
        self.e_rstar = e_rstar
        
        self.prot = prot # hours
        self.e_prot = e_prot
        
        self._upperlimitvsini = upperlimitvsini
        
        
        
        
    def log_likelihood(self, theta):
        
        """
        Calculates the log of the Bayesian likelihood function.
        
        Parameter
        ---------
        theta : list of floats
            Values of the fit parameters. theta[0] is cosi, theta[1] is rstar,
            and theta[2] is prot.
        
        Returns
        -------
        float
            The log-likelihood value.
        
        """
        
        cosi, r, p = theta
        
        
        sini = np.sqrt(1 - cosi**2)
        
        # line-of-sight velocity; km/s
        cv = 2 * np.pi * r * 6.957e5 / (p * 24 * 3600)
        
        cvsini = cv * sini
        
        
        # if vsini is only known as an upper limit
        if self._upperlimitvsini:
            if cvsini > self.vsini:
                return -np.inf
        
            chi2 = (r - self.rstar)**2 / self.e_rstar**2 + (p - self.prot)**2 / self.e_prot**2
            
        else:
            chi2 = (self.vsini - cvsini)**2 / self.e_vsini**2 + (r - self.rstar)**2 / self.e_rstar**2 + (p - self.prot)**2 / self.e_prot**2
        
        
        return -0.5 * chi2
    
    
    
    
    def log_prior(self, theta):
        
        """
        Sets boundaries that act as uniform priors for the fit parameters.
        
        Parameter
        ---------
        theta : list of floats
            Values of the fit parameters. theta[0] is cosi, theta[1] is rstar,
            and theta[2] is prot.
        
        Returns
        -------
        float
            0.0 if the fit parameters lie within the bounds,
            -inf otherwise.
        
        """
        
        cosi, r, p = theta
        
        if cosi < 0 or cosi > 1:
            return -np.inf
        
        if r < 0 or p < 0:
            return -np.inf
        
        
        return 0.0
    
    
    
    
    def log_probability(self, theta):
        
        """
        Calculates the log of the probability as the sum of the log-prior
        and log-likelihood.
        
        Parameter
        ---------
        theta : list of floats
            Values of the fit parameters. theta[0] is cosi, theta[1] is rstar,
            and theta[2] is prot.
        
        Returns
        -------
        float
            The log-probability value (sum of log-prior and log-likelihood). 
        
        """
        
        lp = self.log_prior(theta)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood(theta)
    
    
    
    
class CosI(Probability):
    
    """
    The MCMC estimation and extraction of the posterior distribution for cosi.
    
    Parameters
    ----------
    vsini : float
        Line-of-sight stellar rotation velocity in km/s.
    e_vsini : float
        Uncertainty on vsini in km/s.
    rstar : float
        Stellar radius in solar radii.
    e_rstar : float
        Uncertainty on rstar in solar radii.
    prot : float
        Stellar rotation period in hours.
    e_prot : float
        Uncertainty on prot in hours.
    upperlimitvsini : bool, optional
        Is vsini only available as an upper limit? The default is False,
        i.e., vsini is measured with some confidence.
    
    """
    
    def __init__(self, vsini, e_vsini, rstar, e_rstar, prot, e_prot, upperlimitvsini=False):
        
        self._upperlimitvsini = upperlimitvsini
        
        super().__init__(vsini, e_vsini, rstar, e_rstar, prot, e_prot, upperlimitvsini=self._upperlimitvsini)
        
        
        
        
    def run_mcmc(self, nwalkers, nsteps, position=None, perturbation=0.02, tol=1000, progress=True):
        
        """
        Runs an MCMC simulation using the emcee python package to estimate posterior distributions 
        for the fit parameters cosi, rstar, and prot.
        
        See https://emcee.readthedocs.io/en/stable/ for more information about emcee.
        
        Parameters
        ----------
        nwalkers : int
            The number of ensemble walkers that explore the parameter space.
        nsteps : int
            The total number of iterations.
        position : list of floats, optional
            The unperturbed initial position for each fit parameter. If None (default),
            sets initial positions at 0.5 for cosi and the values used when initializing
            ``class:CosI`` for rstar and prot, respectively.
        perturbation : float, optional
            The positions are varied by `perturbation` * samples of a standard normal
            distribution. The default is 0.02. 
        tol : int, optional
            Number of attempts to initialize walkers within their preferred bounds before an 
            error is raised. The default is 1000.
        progress : bool, optional
            If True (default), displays a progress bar using the tqdm python package.
            
        Returns
        -------
        sampler : emcee.EnsembleSampler
            The MCMC ensemble sampler.
        
        """
        
        ndim = 3
        
        log_prob = self.log_probability
        
        init_pos = position
        
        if init_pos is None:
            init_pos = [0.5, self.rstar, self.prot]
        
            
        pos = []
        i = 0
        while len(pos) < nwalkers:
            
            i+=1
            
            p = init_pos + perturbation * np.random.randn(ndim)
            
            if np.isfinite(log_prob(p)):
                pos.append(p)
                
            if i > tol:
                raise ValueError("Failed to initialize walkers. Try changing the intial position and/or perturbation.")
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        
        if version.parse(emcee.__version__) >= version.parse('3.0.0'):
            sampler.run_mcmc(pos, nsteps, progress=progress)
        
        else:
            warnings.warn("update emcee to at least v3.0.0 or some features may not work")
            sampler.run_mcmc(pos, nsteps)
        
        return sampler
    
    
    
    
    def get_posterior(self, sampler, burnin=500, thin=1):
        
        """
        Extracts the posterior distribution for cosi from the 
        flattened MCMC sampler chain.
        
        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            The the sampler object returned by `func:run_mcmc`.
        burnin : int, optional
            The number of steps to discard from the beginning of the posterior
            chain. The default is 500 steps or a third of the total number of steps
            rounded to the nearest integer if the number of steps is less than 1500.
        thin : int, optional
            Only take every `thin` amount from the flattened chain. The default is 1.
        
        """
        
        if len(sampler.get_chain()) < 1500:
            burnin = len(sampler.get_chain()) // 3
        
        chain = sampler.get_chain(flat=True, discard=burnin, thin=thin)
        
        return chain[:, 0]
    
    
    
    
    

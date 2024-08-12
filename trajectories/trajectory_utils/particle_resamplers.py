import numpy as np
from numpy.random import default_rng
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class ParticleResampler(ABC):
    def __init__(self, name: str, seed: int) -> None:
        """Abstact base class for particle resamplers

        Args:
            name (str): Name of resampler
            seed (int): Internal seed
        """
        super().__init__()
        self._name = name
        self._seed = seed
        self._rng = default_rng(self._seed)
        self._smallest = np.nextafter(np.float64(0), np.float64(1))

    def resample(self, x_p: np.ndarray, y: np.ndarray, weights_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resample new particles for the next step given the new observation and probability of last particles

        Args:
            x_p (np.ndarray): Particles
            y (np.ndarray): Current observation (as 2-D array of shape [points, xy])
            weights_prev (np.ndaray): Previous weight of each particle

        Returns:
            Tuple[np.ndarray, np.ndarray]: Array of indices for new particles,  weight for each particle
        """
        if len(y) > 0:
            weights = self._judge_particle(x_p, y) * weights_prev
            weights = weights / np.maximum(np.sum(weights), self._smallest)
        else:
            weights = np.full(x_p.shape[0], 1 / x_p.shape[0])
        
        idx = self._rng.choice(x_p.shape[0], size=x_p.shape[0], p=weights)
        # ret_idxs = np.arange(len(idx))
        # to_resample = np.where(weights < 0.001)
        # ret_idxs[to_resample] = idx[to_resample]
        # 
        # return ret_idxs, p_new, weights
        return idx, weights

    @property
    def name(self) -> str:
        """Get the name of this merger

        Returns:
            str: Name of merger
        """
        return self._name
    
    @abstractmethod
    def _judge_particle(self, x_p: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the probability of each particle given the new observation

        Args:
            x_p (np.ndarray): Particles
            y (np.ndarray): New observation

        Returns:
            np.ndarray: Probability of each particle
        """
        pass
    

class L2ParticleResampler(ParticleResampler):
    def __init__(self, seed: int = 42) -> None:
        """Judge paticles based on L2 distance to the observation. \
            The distances will be normalized by the maximum occuring distance and the subtracted from 1

        Args:
            seed (int, optional): Seed of internal random state. Defaults to 42.
        """
        super().__init__("L2ParticleResampler", seed)
        
    def _judge_particle(self, x_p: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(y) == 0:
            return np.ones(x_p.shape[0]) / x_p.shape[0]
        dist = np.linalg.norm(np.repeat(x_p[:, np.newaxis, :], repeats=y.shape[0], axis=1) - y, axis=-1)
        dist = np.min(dist, axis=-1)
        dist -= np.min(dist)
        prob_inv = dist / np.max(dist)
        return 1 / np.maximum(prob_inv, 1e-4)
    
class SoftmaxParticleResampler(ParticleResampler):
    def __init__(self, seed: int = 42) -> None:
        """Judge paticles based on the L2 distance + softmax to the observation. 

        Args:
            seed (int, optional): Seed of internal random state. Defaults to 42.
        """
        super().__init__("L2ParticleResampler", seed)
        
    def _judge_particle(self, x_p: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(y) == 0:
            return np.ones(x_p.shape[0]) / x_p.shape[0]
        dist = np.linalg.norm(np.repeat(x_p[:, np.newaxis, :], repeats=y.shape[0], axis=1) - y, axis=-1)
        dist = np.min(dist, axis=-1)
        # min_dist = np.min(dist)
        # if np.min(dist) > 400:
        #     dist -= min_dist
        dist = np.maximum(np.exp(- dist), self._smallest)
        dist /= np.sum(dist)
        return dist
        
        
class GaussianParticleResampler(ParticleResampler):
    def __init__(self, std: float, seed: int = 42) -> None:
        """Gaussian Particle resampler will judge the distance of particles to the observation based on a gaussian distance

        Args:
            std (float): Standard deviation of gaussian distance
            seed (int, optional): Seed for internal random state. Defaults to 42.
        """
        self._std = std
        self._gaussian_scaling = 1 / (self._std * np.sqrt(2 * np.pi))
        super().__init__("GaussianParticleResampler", seed)
        
    def _judge_particle(self, x_p: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(y) == 0:
            return np.ones(x_p.shape[0]) / x_p.shape[0]
        dist = np.linalg.norm(np.repeat(x_p[:, np.newaxis, :], repeats=y.shape[0], axis=1) - y, axis=-1)
        dist = np.min(dist, axis=-1)
        dist -= np.min(dist)
        return np.exp(-0.5 * (dist / self._std) ** 2) * self._gaussian_scaling
        
PARTICLE_RESAMPLER_MAP : Dict[str, ParticleResampler] = {
    "gauss": GaussianParticleResampler,
    "softmax": SoftmaxParticleResampler,
    "l2": L2ParticleResampler
}
        
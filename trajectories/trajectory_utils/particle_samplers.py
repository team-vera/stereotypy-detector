import numpy as np
from numpy.random import default_rng, Generator
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from utilities.enums import MAX_SPEED_PB_PIXELS


class ParticleSampler(ABC):
    def __init__(self, name: str, seed: int) -> None:
        """Abstact base class for particle samplers

        Args:
            name (str): Name of sampler
            seed (int): Internal seed
        """
        super().__init__()
        self._name = name
        self._seed = seed
        self._rng = default_rng(self._seed)

    @abstractmethod
    def sample(self, 
               x: np.ndarray, 
               prev_delta_x: np.ndarray = None, 
               y: np.ndarray = None) -> np.ndarray:
        """Sample a new particle for each given previous state given the observation

        Args:
            x (np.ndarray): Previous states
            prev_delta_x (np.ndarray, optional): Previous state movement vector. Defaults to None.
            y (np.ndarray, optional): Observation of the curent time step. Defaults to None.

        Returns:
            np.ndarray: Array of particles
        """
        pass

    @property
    def name(self) -> str:
        """Get the name of this merger

        Returns:
            str: Name of merger
        """
        return self._name
    
    @property
    def rng(self) -> Generator:
        """Default internal random number generator

        Returns:
            Generator: Default internal random number generator
        """
        return self._rng
    
    def _to_cart(self, r: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """Convert polar to cartesian coordiantes

        Args:
            r (np.ndarray): Radii
            angle (np.ndarray): Anlges

        Returns:
            np.ndarray: Cartesian coordiantes
        """
        xy = np.ndarray((r.shape[0], 2), dtype=float)
        xy[:, 0] = r * np.cos(angle)
        xy[:, 1] = r * np.sin(angle)
        return xy
    
    def reindex(self, idxs: np.ndarray):
        """Reindex internal memory

        Args:
            idxs (np.ndarray): Indices to reindex to
        """
        pass
    
    def reset(self) -> None:
        """Reset internal memory"""
        pass
    
    def reset_random(self) -> None:
        """Reset internal memory to random values"""
        pass

class NaiveParticleSampler(ParticleSampler):
    def __init__(self, std_radius: float, seed: int = 42) -> None:
        super().__init__("NaiveParticleSampler", seed)
        self._std_radius = std_radius
        self._var_radius = std_radius ** 2
        self._r_gen = default_rng(self._seed)
        self._angle_gen = default_rng(self._seed)
        
    def sample(self, x: np.ndarray, prev_delta_x: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        r = self._r_gen.normal(scale=self._std_radius, size=x.shape[0])
        angle = self._angle_gen.random(x.shape[0]) * np.pi
        cart = self._to_cart(r, angle)
        # probability for radius (gaussian) x probability for angle (uniform from 0 to pi)
        # prob = np.exp(- (r ** 2) / (2 * self._var_radius)) / \
        #     (np.sqrt(2 * np.pi * self._var_radius ** 2) * np.pi)
        return x + prev_delta_x + cart
    
class NoiseParticleSampler(ParticleSampler):
    def __init__(self, std_radius: float, seed: int = 42) -> None:
        super().__init__("NoiseParticleSampler", seed)
        self._std_radius = std_radius
        self._var_radius = std_radius ** 2
        self._r_gen = default_rng(self._seed)
        self._angle_gen = default_rng(self._seed)
        
    def sample(self, x: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        r = self._r_gen.normal(scale=self._std_radius, size=x.shape[0])
        angle = self._angle_gen.random(x.shape[0]) * np.pi
        cart = self._to_cart(r, angle)
        # probability for radius (gaussian) x probability for angle (uniform from 0 to pi)
        # prob = np.exp(- (r ** 2) / (2 * self._var_radius)) / \
        #     (np.sqrt(2 * np.pi * self._var_radius ** 2) * np.pi)
        return x + cart
        
class ObservationParticleSampler(ParticleSampler):
    def __init__(self, std_radius: float, seed: int = 42) -> None:
        super().__init__("ObservationParticleSampler", seed)
        self._std_radius = std_radius
        self._var_radius = std_radius ** 2
        self._r_gen = default_rng(self._seed)
        self._angle_gen = default_rng(self._seed)
        self._y_selector = default_rng(self._seed)
        
    def sample(self, x: np.ndarray, y: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        r = self._r_gen.normal(scale=self._std_radius, size=x.shape[0])
        angle = self._angle_gen.random(x.shape[0]) * np.pi
        cart = self._to_cart(r, angle)
        # probability for radius (gaussian) x probability for angle (uniform from 0 to pi)
        # prob = np.exp(- (r ** 2) / (2 * self._var_radius)) / \
        #     (np.sqrt(2 * np.pi * self._var_radius ** 2) * np.pi)
        if len(y) == 1:
            return y + cart
        if len(y) > 1:
            y_basis = self._y_selector.choice(y, size=cart.shape[0])
            # prob /= len(y)
            return y_basis + cart
        else:
            return x + cart
        
class UniformParticleSampler(ParticleSampler):
    def __init__(self, radius: float, seed: int = 42) -> None:
        super().__init__("NoiseParticleSampler", seed)
        self._radius = radius
        self._r_gen = default_rng(self._seed)
        self._angle_gen = default_rng(self._seed)
        #self._prob = 1 / (radius * np.pi)
        
    def sample(self, x: np.ndarray, prev_delta_x: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        r = self._r_gen.random(x.shape[0]) * self._radius
        angle = self._angle_gen.random(x.shape[0]) * np.pi
        cart = self._to_cart(r, angle)
        return x + prev_delta_x + cart #, np.full(x.shape[0], self._prob)
        
        
class MomentumParticleSampler(ParticleSampler):
    def __init__(self, std_radius: float, momentum_weight: float = 0.9, seed: int = 42) -> None:
        super().__init__("MomentumParticleSampler", seed)
        self._std_radius = std_radius
        self._var_radius = std_radius ** 2
        self._r_gen = default_rng(self._seed)
        self._angle_gen = default_rng(self._seed)
        self._momentum_weight = momentum_weight
        self._new_movement_weight = 1 - momentum_weight
        self._momentum = None
        
    def sample(self, x: np.ndarray, prev_delta_x: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if self._momentum is None:
            self._momentum = prev_delta_x
        r = self._r_gen.normal(scale=self._std_radius, size=x.shape[0])
        angle = self._angle_gen.random(x.shape[0]) * np.pi
        cart = self._to_cart(r, angle)
        # probability for radius (gaussian) x probability for angle (uniform from 0 to pi)
        # prob = np.exp(- (r ** 2) / (2 * self._var_radius)) / \
        #     (np.sqrt(2 * np.pi * self._var_radius ** 2) * np.pi)
        self._momentum = self._new_movement_weight * prev_delta_x + self._momentum_weight * self._momentum
        return x + self._momentum + cart
    
    def reindex(self, idxs: np.ndarray):
        self._momentum = self._momentum[idxs]
        reset_momentum = self._momentum.shape[0] // 20
        self._momentum[:reset_momentum] = np.zeros_like((reset_momentum, 2))
        
    def reset(self) -> None:
        self._momentum = None
        
    def reset_random(self) -> None:
        self._momentum = 2 * self._rng.random(self._momentum.shape) - 1
        self._momentum *= np.sqrt(MAX_SPEED_PB_PIXELS)
        
PARTICLE_SAMPLER_MAP: Dict[str, ParticleSampler] = {
    "naive": NaiveParticleSampler,
    "noise": NoiseParticleSampler,
    "observation": ObservationParticleSampler,
    "uniform": UniformParticleSampler,
    "momentum": MomentumParticleSampler
}
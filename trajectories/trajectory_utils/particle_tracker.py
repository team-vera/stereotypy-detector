import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class ParticleTracker(ABC):
    def __init__(self, name: str, num_particles: int) -> None:
        """Abstact base class for particle tracker

        Args:
            name (str): Name of sampler
            num_particles (int): Number of particles
        """
        super().__init__()
        self._name = name
        self._num_particles = num_particles
        self._trajectory_length = None
        self._trajectory = None
        self._internal_counter = 0
        self._invalid = - np.ones(2)
        

    @abstractmethod
    def add_particles(self, x_p: np.ndarray, idxs: np.ndarray, weights: np.ndarray):
        """Add the next particles

        Args:
            x_p (np.ndarray): New particles
            idxs (np.ndarray): Indices of resampling step
            weights (np.ndarray): Weights for each particle
        """
        pass
    
    def reset(self, trajectory_length: int):
        self._trajectory_length = trajectory_length
        self._trajectory = np.full((trajectory_length, 2), self._invalid)
        self._internal_counter = 0

    @property
    def name(self) -> str:
        """Get the name of this tracker

        Returns:
            str: Name of merger
        """
        return self._name
    
    @property
    def trajectory(self) -> np.ndarray:
        return self._trajectory
    
    def __setitem__(self, idx: int, value: np.ndarray):
        self._trajectory[idx] = value
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return self._trajectory[idx]
    
    def set_index(self, idx: int):
        self._internal_counter = idx

class DelayParticleTracker(ParticleTracker):
    def __init__(self, 
                 num_particles: int,
                 memory_size: int) -> None:
        super().__init__("DelayParticleTracker", num_particles)
        self._memory_size = memory_size
        #self._index_memory = np.full((memory_size, num_particles), 0)
        self._particle_memory = np.full((memory_size, num_particles, 2), self._invalid)
        self._memory_idx = 0
        
    def reset(self, trajectory_length: int):
        super().reset(trajectory_length)
        #self._index_memory = np.full((self._memory_size, self._num_particles), 0)
        self._particle_memory = np.full((self._memory_size, self._num_particles, 2), self._invalid)
        self._memory_idx = 0
        
    def add_particles(self, 
                      x_p: np.ndarray, 
                      idxs: np.ndarray, 
                      weights: np.ndarray):
        # set trajectory point for intermediate processing
        self._trajectory[self._internal_counter] = np.sum(x_p * weights[:, np.newaxis], axis=0)
        
        local_memory_idx = self._memory_idx % self._memory_size
        
        mean_memory = np.mean(self._particle_memory, axis=1)
        
        if self._memory_idx >= self._memory_size:   
            self._trajectory[self._internal_counter - self._memory_size:self._internal_counter - local_memory_idx] = mean_memory[local_memory_idx:]
        if self._memory_idx != 0:
            self._trajectory[self._internal_counter - local_memory_idx:self._internal_counter] = mean_memory[:local_memory_idx]
                    
            
        self._particle_memory[local_memory_idx] = x_p
        self._particle_memory = self._particle_memory[:, idxs]
        
        #self._index_memory = self._index_memory[:, idxs]
        
        self._internal_counter += 1
        self._memory_idx += 1
        
    @property
    def trajectory(self) -> np.ndarray:
        #for i in range(min(self._memory_size, self._memory_idx)):
        #    #self._trajectory[-i] = np.mean(self._particle_memory[i, self._index_memory[i]], axis=0)
        #    self._trajectory[-i] = np.mean(self._particle_memory[i], axis=0)
        return super().trajectory
    

class DummyMeanParticleTracker(ParticleTracker):
    def __init__(self, 
                 num_particles: int) -> None:
        super().__init__("DummyMeanParticleTracker", num_particles)
        
    def add_particles(self, 
                      x_p: np.ndarray, 
                      idxs: np.ndarray,
                      weights: np.ndarray):
        self._trajectory[self._internal_counter] = np.sum(x_p * weights[:, np.newaxis], axis=0)
        self._internal_counter += 1
        
PARTICLE_TRACKER_MAP: Dict[str, ParticleTracker] = {
    "dummy": DummyMeanParticleTracker,
    "delay": DelayParticleTracker
}   
        
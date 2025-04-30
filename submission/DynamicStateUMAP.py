import numpy as np
import umap
import os
import pickle
from pathlib import Path

class DynamicStateUMAP:
    """
    A class for reducing the dimensionality of neural state data using UMAP.
    This replaces the PCA-based state reduction in the DQN implementation.
    """
    
    def __init__(self, state_dim=4, model_path=None):
        """
        Initialize the UMAP-based state reducer.
        
        Args:
            state_dim: The dimension of the reduced state (default: 4)
            model_path: Path to a pre-trained model to load (optional)
        """
        self.state_dim = state_dim
        self.reducer = None
        self.is_trained = False
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def prepare_data(self, spikes, elecs):
        """
        Convert spike and electrode data to a binary matrix format.
        
        Args:
            spikes: List of spike time arrays
            elecs: List of electrode arrays
            
        Returns:
            Binary matrix of shape (n_samples, n_elecs * n_time_bins)
        """
        print(f"Input data shapes - spikes: {len(spikes)}, elecs: {len(elecs)}")
        
        # Set default dimensions in case all arrays are empty
        max_elec = 4  # Default number of electrodes
        max_time = 20  # Default time bins
        
        # Determine dimensions from the first non-empty sample
        for i in range(len(spikes)):
            if len(spikes[i]) > 0:
                print(f"Sample {i} - spikes shape: {spikes[i].shape}, elecs shape: {elecs[i].shape}")
                # Find the maximum electrode index and time bin
                max_elec = max(max_elec, int(np.max(elecs[i])) + 1)
                max_time = max(max_time, int(np.max(spikes[i])) + 1)
        
        print(f"Final dimensions - max_elec: {max_elec}, max_time: {max_time}")
        
        # Create binary matrix
        X = np.zeros((len(spikes), max_elec * max_time))
        
        for i in range(X.shape[0]):
            if len(spikes[i]) == 0:
                continue
            for j in range(spikes[i].shape[0]):
                X[i, int(elecs[i][j] * max_time + spikes[i][j])] = 1
        
        print(f"Output matrix shape: {X.shape}")
        return X
    
    def train(self, spikes, elecs, n_neighbors=15, min_dist=0.1, save_path=None):
        """
        Train the UMAP reducer on the provided neural data.
        
        Args:
            spikes: List of spike time arrays
            elecs: List of electrode arrays
            n_neighbors: Number of neighbors for UMAP (default: 15)
            min_dist: Minimum distance for UMAP (default: 0.1)
            save_path: Path to save the trained model (optional)
            
        Returns:
            The reduced data matrix
        """
        # Prepare data
        X = self.prepare_data(spikes, elecs)
        
        # Check if we have valid data
        if X.shape[0] == 0:
            print("Warning: No valid data to train on.")
            return None
        
        # Create and fit UMAP reducer
        self.reducer = umap.UMAP(
            n_components=self.state_dim,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        
        # Fit and transform the data
        X_reduced = self.reducer.fit_transform(X)
        
        self.is_trained = True
        
        # Save model if path is provided
        if save_path:
            self.save_model(save_path)
        
        return X_reduced
    
    def get_state(self, response):
        """
        Reduce the dimensionality of neural state data.
        
        Args:
            response: Array of shape (n_spikes, 2) where first column is time and second is electrode
            
        Returns:
            Reduced state vector of dimension state_dim
        """
        if not self.is_trained or self.reducer is None:
            return np.zeros(self.state_dim)
        
        # Prepare data using the same method as in prepare_data
        max_elec = 4  # Default number of electrodes
        max_time = 20  # Default time bins
        
        # Determine dimensions from the response
        if response.shape[0] > 0:
            max_elec = max(max_elec, int(np.max(response[:, 1])) + 1)
            max_time = max(max_time, int(np.max(response[:, 0])) + 1)
        
        # Create binary matrix with fixed size
        n_features = max_elec * max_time
        X = np.zeros((1, n_features))
        
        if response.shape[0] > 0:
            for j in range(response.shape[0]):
                # Ensure indices are within bounds
                elec_idx = min(int(response[j, 1]), max_elec - 1)
                time_idx = min(int(response[j, 0]), max_time - 1)
                feature_idx = elec_idx * max_time + time_idx
                if feature_idx < n_features:  # Additional safety check
                    X[0, feature_idx] = 1
        
        # Get reduced state
        reduced_state = self.reducer.transform(X)[0]
        
        return reduced_state
    
    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        with open(path, 'wb') as f:
            pickle.dump({
                'reducer': self.reducer,
                'state_dim': self.state_dim,
                'is_trained': self.is_trained
            }, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.reducer = state['reducer']
        self.state_dim = state['state_dim']
        self.is_trained = state['is_trained']
        
        print(f"Model loaded from {path}")
    
    def print_info(self):
        """
        Print detailed information about the state reduction model
        """
        print(f"\n===== STATE REDUCTION MODEL INFO =====")
        print(f"State dimension: {self.state_dim}")
        if self.reducer is not None:
            print(f"UMAP parameters:")
            print(f"  n_neighbors: {self.reducer.n_neighbors}")
            print(f"  min_dist: {self.reducer.min_dist}")
            print(f"  n_components: {self.reducer.n_components}")
        print(f"Model trained: {self.is_trained}")
        print("=======================================") 
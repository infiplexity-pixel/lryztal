import torch
import numpy as np
from typing import Dict, Tuple, Type, Optional, List
from collections import OrderedDict
import re

class GeneralModuleSerializer:
    """
    Universal serializer/deserializer for any PyTorch model with exclusion support.
    
    Example:
        # Serialize with exclusions
        serializer = GeneralModuleSerializer(chunk_size=128)
        dataset, metadata = serializer.serialize(
            model, 
            exclude_patterns=['embedding', 'embed'],  # Exclude by name patterns
            exclude_names=['layer.0.weight']          # Or exact names
        )
        
        # Deserialize (only non-excluded params will be loaded)
        model = serializer.deserialize(dataset, metadata, MyModelClass)
    """
    
    def __init__(self, chunk_size: int = 128):
        """
        Args:
            chunk_size: The 'i' dimension in (N, i) output format
        """
        self.chunk_size = chunk_size
        
    def _should_exclude(self, param_name: str, 
                       exclude_patterns: Optional[List[str]] = None,
                       exclude_names: Optional[List[str]] = None) -> bool:
        """
        Check if a parameter should be excluded based on patterns or exact names.
        
        Args:
            param_name: Name of the parameter
            exclude_patterns: List of regex patterns to match against param name
            exclude_names: List of exact names to exclude
        
        Returns:
            bool: True if parameter should be excluded
        """
        # Check exact name matches
        if exclude_names:
            if param_name in exclude_names:
                return True
        
        # Check pattern matches
        if exclude_patterns:
            for pattern in exclude_patterns:
                if re.search(pattern, param_name):
                    return True
        
        return False
    
    def serialize(self, 
                 model: torch.nn.Module,
                 exclude_patterns: Optional[List[str]] = None,
                 exclude_names: Optional[List[str]] = None,
                 include_only_patterns: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Convert any PyTorch model to flat dataset (N, chunk_size)
        
        Args:
            model: Any PyTorch model
            exclude_patterns: List of regex patterns to exclude parameters (e.g., ['embedding', 'bias'])
            exclude_names: List of exact parameter names to exclude
            include_only_patterns: List of regex patterns - if provided, ONLY these params are included
            
        Returns:
            dataset: numpy array of shape (total_params//chunk_size, chunk_size)
            metadata: Dictionary with parameter shapes and names
        """
        state_dict = model.state_dict()
        param_names = []
        param_shapes = []
        all_params = []
        
        # Determine inclusion/exclusion
        for name, param in state_dict.items():
            # Check if we should include this parameter
            should_include = True
            
            # If include_only_patterns is provided, only include matching params
            if include_only_patterns:
                should_include = False
                for pattern in include_only_patterns:
                    if re.search(pattern, name):
                        should_include = True
                        break
            
            # Check exclusions (only if include_only_patterns didn't already exclude it)
            if should_include and self._should_exclude(name, exclude_patterns, exclude_names):
                should_include = False
            
            if should_include:
                param_flat = param.detach().cpu().numpy().flatten()
                all_params.append(param_flat)
                param_names.append(name)
                param_shapes.append(param.shape)
            else:
                # Store info about excluded parameters but don't serialize their values
                # We'll keep track of them in metadata
                print(f"Excluding parameter: {name} (shape: {param.shape})")
        
        # Store excluded parameters info
        excluded_params = []
        for name, param in state_dict.items():
            if name not in param_names:
                excluded_params.append({
                    'name': name,
                    'shape': param.shape,
                    'is_excluded': True
                })
        
        if not all_params:
            raise ValueError("No parameters were selected for serialization. Check your inclusion/exclusion patterns.")
        
        # Concatenate all parameters
        all_params_flat = np.concatenate(all_params)
        total_params = len(all_params_flat)
        
        # Pad to make divisible by chunk_size
        padding_needed = (self.chunk_size - (total_params % self.chunk_size)) % self.chunk_size
        if padding_needed > 0:
            all_params_flat = np.pad(all_params_flat, (0, padding_needed), mode='constant')
        
        # Reshape to (N, chunk_size)
        num_rows = len(all_params_flat) // self.chunk_size
        dataset = all_params_flat.reshape(num_rows, self.chunk_size)
        
        # Metadata for reconstruction
        metadata = {
            'param_names': param_names,
            'param_shapes': param_shapes,
            'total_params': total_params,
            'padding_needed': padding_needed,
            'chunk_size': self.chunk_size,
            'num_rows': num_rows,
            'excluded_params': excluded_params,  # Store excluded params info
            'excluded_patterns': exclude_patterns,
            'excluded_names': exclude_names,
            'include_only_patterns': include_only_patterns
        }
        
        return dataset, metadata
    
    def deserialize(self, 
                   dataset: np.ndarray, 
                   metadata: Dict, 
                   model_class: Type[torch.nn.Module],
                   **model_kwargs) -> torch.nn.Module:
        """
        Reconstruct model from dataset and metadata
        
        Args:
            dataset: numpy array from serialize()
            metadata: metadata dictionary from serialize()
            model_class: The model class to instantiate (e.g., MyModel)
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            model: Reconstructed PyTorch model with updated non-excluded parameters
        """
        # Flatten dataset
        flat_params = dataset.flatten()
        
        # Remove padding
        if metadata.get('padding_needed', 0) > 0:
            flat_params = flat_params[:-metadata['padding_needed']]
        
        # Create model instance
        model = model_class(**model_kwargs) if model_kwargs else model_class()
        
        # Get current state dict (will contain all parameters)
        current_state_dict = model.state_dict()
        
        # Create new state dict by merging serialized and existing parameters
        new_state_dict = OrderedDict()
        idx = 0
        
        # First, copy all existing parameters
        for name, param in current_state_dict.items():
            new_state_dict[name] = param.clone()
        
        # Then update with serialized parameters
        for name, shape in zip(metadata['param_names'], metadata['param_shapes']):
            param_size = np.prod(shape)
            param_flat = flat_params[idx:idx + param_size]
            idx += param_size
            
            # Reshape and convert to tensor
            param_reshaped = param_flat.reshape(shape)
            new_state_dict[name] = torch.from_numpy(param_reshaped).float()
        
        # Load parameters (this will update only the parameters in the state dict)
        model.load_state_dict(new_state_dict, strict=False)  # strict=False allows missing params
        
        return model

    def serialize_to_files(self, 
                          model: torch.nn.Module, 
                          dataset_path: str, 
                          metadata_path: str,
                          exclude_patterns: Optional[List[str]] = None,
                          exclude_names: Optional[List[str]] = None,
                          include_only_patterns: Optional[List[str]] = None) -> None:
        """Convenience: serialize and save to files"""
        dataset, metadata = self.serialize(
            model, 
            exclude_patterns=exclude_patterns,
            exclude_names=exclude_names,
            include_only_patterns=include_only_patterns
        )
        np.save(dataset_path, dataset)
        torch.save(metadata, metadata_path)
    
    def deserialize_from_files(self,
                              dataset_path: str,
                              metadata_path: str,
                              model_class: Type[torch.nn.Module],
                              **model_kwargs) -> torch.nn.Module:
        """Convenience: load from files and deserialize"""
        dataset = np.load(dataset_path)
        metadata = torch.load(metadata_path)
        return self.deserialize(dataset, metadata, model_class, **model_kwargs)

    def get_excluded_params_info(self, metadata: Dict) -> List[Dict]:
        """Get information about excluded parameters"""
        return metadata.get('excluded_params', [])
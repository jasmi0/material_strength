"""
Material Optimization Engine for Material Strength & Selection System
Handles material selection based on user constraints and predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MaterialOptimizer:
    """
    Optimizer for selecting the best materials based on constraints and predictions.
    """
    
    def __init__(self, materials_data: pd.DataFrame, model_manager=None):
        """
        Initialize the material optimizer.
        
        Args:
            materials_data (pd.DataFrame): Original materials data
            model_manager: Trained model manager for predictions
        """
        self.materials_data = materials_data.copy()
        self.model_manager = model_manager
        self.feasible_materials = None
        self.optimization_results = None
    
    def set_constraints(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set constraints for material selection.
        
        Args:
            constraints (Dict): Dictionary of constraints
                - max_cost: Maximum cost per kg (USD)
                - min_strength: Minimum tensile strength (MPa)
                - min_availability: Minimum availability score (0-10)
                - min_environmental: Minimum environmental impact score (0-10)
                - material_types: List of acceptable material types
                - max_density: Maximum density (kg/m³)
                - min_corrosion_resistance: Minimum corrosion resistance score
                
        Returns:
            Dict: Processed constraints with defaults
        """
        default_constraints = {
            'max_cost': float('inf'),
            'min_strength': 0,
            'min_availability': 0,
            'min_environmental': 0,
            'material_types': None,
            'max_density': float('inf'),
            'min_corrosion_resistance': 0,
            'max_environmental_impact': 10,  # Lower environmental impact is better
            'min_yield_strength': 0,
            'min_elastic_modulus': 0
        }
        
        # Update with user constraints
        processed_constraints = default_constraints.copy()
        processed_constraints.update(constraints)
        
        # Validate constraints
        for key, value in processed_constraints.items():
            if key in ['max_cost', 'min_strength', 'min_availability', 'max_density'] and value < 0:
                raise ValueError(f"Constraint {key} must be non-negative")
        
        return processed_constraints
    
    def filter_by_constraints(self, constraints: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter materials based on constraints.
        
        Args:
            constraints (Dict): Material constraints
            
        Returns:
            pd.DataFrame: Filtered materials that meet constraints
        """
        filtered_data = self.materials_data.copy()
        
        # Apply cost constraint
        if constraints['max_cost'] != float('inf'):
            filtered_data = filtered_data[filtered_data['cost_per_kg_usd'] <= constraints['max_cost']]
        
        # Apply strength constraint
        if constraints['min_strength'] > 0:
            filtered_data = filtered_data[filtered_data['tensile_strength_mpa'] >= constraints['min_strength']]
        
        # Apply availability constraint
        if constraints['min_availability'] > 0:
            filtered_data = filtered_data[filtered_data['availability_score'] >= constraints['min_availability']]
        
        # Apply environmental constraint
        if constraints['min_environmental'] > 0:
            filtered_data = filtered_data[filtered_data['environmental_impact_score'] >= constraints['min_environmental']]
        
        # Apply max environmental impact constraint (lower is better)
        if constraints['max_environmental_impact'] < 10:
            filtered_data = filtered_data[filtered_data['environmental_impact_score'] <= constraints['max_environmental_impact']]
        
        # Apply material type constraint
        if constraints['material_types'] is not None:
            filtered_data = filtered_data[filtered_data['material_type'].isin(constraints['material_types'])]
        
        # Apply density constraint
        if constraints['max_density'] != float('inf'):
            filtered_data = filtered_data[filtered_data['density_kg_m3'] <= constraints['max_density']]
        
        # Apply corrosion resistance constraint
        if constraints['min_corrosion_resistance'] > 0:
            filtered_data = filtered_data[filtered_data['corrosion_resistance_score'] >= constraints['min_corrosion_resistance']]
        
        # Apply yield strength constraint
        if constraints['min_yield_strength'] > 0:
            filtered_data = filtered_data[filtered_data['yield_strength_mpa'] >= constraints['min_yield_strength']]
        
        # Apply elastic modulus constraint
        if constraints['min_elastic_modulus'] > 0:
            filtered_data = filtered_data[filtered_data['elastic_modulus_gpa'] >= constraints['min_elastic_modulus']]
        
        self.feasible_materials = filtered_data
        return filtered_data
    
    def calculate_material_scores(self, materials: pd.DataFrame, 
                                weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Calculate composite scores for materials based on multiple criteria.
        
        Args:
            materials (pd.DataFrame): Materials to score
            weights (Dict): Weights for different criteria
            
        Returns:
            pd.DataFrame: Materials with calculated scores
        """
        if weights is None:
            weights = {
                'strength': 0.3,
                'cost': 0.25,
                'availability': 0.2,
                'environmental': 0.15,
                'corrosion_resistance': 0.1
            }
        
        # Validate weights sum to 1
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        scored_materials = materials.copy()
        
        # Normalize criteria to 0-1 scale
        # Higher is better for strength, availability, environmental, corrosion resistance
        # Lower is better for cost
        
        if len(materials) > 1:
            # Strength score (higher is better)
            strength_range = materials['tensile_strength_mpa'].max() - materials['tensile_strength_mpa'].min()
            if strength_range > 0:
                scored_materials['strength_score'] = (
                    (materials['tensile_strength_mpa'] - materials['tensile_strength_mpa'].min()) / strength_range
                )
            else:
                scored_materials['strength_score'] = 1.0
            
            # Cost score (lower is better, so invert)
            cost_range = materials['cost_per_kg_usd'].max() - materials['cost_per_kg_usd'].min()
            if cost_range > 0:
                scored_materials['cost_score'] = 1 - (
                    (materials['cost_per_kg_usd'] - materials['cost_per_kg_usd'].min()) / cost_range
                )
            else:
                scored_materials['cost_score'] = 1.0
            
            # Availability score (already 0-10, normalize to 0-1)
            scored_materials['availability_score_norm'] = materials['availability_score'] / 10
            
            # Environmental score (already 0-10, normalize to 0-1)
            scored_materials['environmental_score_norm'] = materials['environmental_impact_score'] / 10
            
            # Corrosion resistance score (already 0-10, normalize to 0-1)
            scored_materials['corrosion_score_norm'] = materials['corrosion_resistance_score'] / 10
        else:
            # Single material case
            scored_materials['strength_score'] = 1.0
            scored_materials['cost_score'] = 1.0
            scored_materials['availability_score_norm'] = materials['availability_score'] / 10
            scored_materials['environmental_score_norm'] = materials['environmental_impact_score'] / 10
            scored_materials['corrosion_score_norm'] = materials['corrosion_resistance_score'] / 10
        
        # Calculate composite score
        scored_materials['composite_score'] = (
            weights['strength'] * scored_materials['strength_score'] +
            weights['cost'] * scored_materials['cost_score'] +
            weights['availability'] * scored_materials['availability_score_norm'] +
            weights['environmental'] * scored_materials['environmental_score_norm'] +
            weights['corrosion_resistance'] * scored_materials['corrosion_score_norm']
        )
        
        return scored_materials
    
    def optimize_material_selection(self, constraints: Dict[str, Any], 
                                  weights: Dict[str, float] = None,
                                  top_n: int = 10) -> Dict[str, Any]:
        """
        Optimize material selection based on constraints and weights.
        
        Args:
            constraints (Dict): Material constraints
            weights (Dict): Weights for different criteria
            top_n (int): Number of top materials to return
            
        Returns:
            Dict: Optimization results
        """
        print("Starting material optimization...")
        
        # Process constraints
        processed_constraints = self.set_constraints(constraints)
        
        # Filter materials by constraints
        feasible_materials = self.filter_by_constraints(processed_constraints)
        
        if len(feasible_materials) == 0:
            return {
                'status': 'no_feasible_materials',
                'message': 'No materials satisfy the given constraints',
                'recommended_materials': pd.DataFrame(),
                'constraints': processed_constraints,
                'total_materials': len(self.materials_data),
                'feasible_count': 0
            }
        
        # Calculate scores
        scored_materials = self.calculate_material_scores(feasible_materials, weights)
        
        # Sort by composite score (descending)
        top_materials = scored_materials.sort_values('composite_score', ascending=False).head(top_n)
        
        # Prepare results
        optimization_results = {
            'status': 'success',
            'message': f'Found {len(feasible_materials)} feasible materials',
            'recommended_materials': top_materials,
            'constraints': processed_constraints,
            'weights': weights,
            'total_materials': len(self.materials_data),
            'feasible_count': len(feasible_materials),
            'top_material': {
                'name': top_materials.iloc[0]['material_name'],
                'type': top_materials.iloc[0]['material_type'],
                'score': top_materials.iloc[0]['composite_score'],
                'strength': top_materials.iloc[0]['tensile_strength_mpa'],
                'cost': top_materials.iloc[0]['cost_per_kg_usd']
            } if len(top_materials) > 0 else None
        }
        
        self.optimization_results = optimization_results
        print(f"Optimization completed. Top material: {optimization_results['top_material']['name'] if optimization_results['top_material'] else 'None'}")
        
        return optimization_results
    
    def get_material_recommendations(self, application_type: str) -> Dict[str, Any]:
        """
        Get material recommendations based on application type.
        
        Args:
            application_type (str): Type of engineering application
            
        Returns:
            Dict: Application-specific recommendations
        """
        application_constraints = {
            'structural': {
                'min_strength': 300,
                'min_yield_strength': 200,
                'max_cost': 10.0,
                'min_availability': 7
            },
            'aerospace': {
                'min_strength': 500,
                'max_density': 3000,
                'max_cost': 50.0,
                'min_environmental': 6,
                'material_types': ['Titanium', 'Aluminum', 'Composite']
            },
            'automotive': {
                'min_strength': 250,
                'max_cost': 5.0,
                'min_availability': 8,
                'max_density': 8000
            },
            'marine': {
                'min_strength': 200,
                'min_corrosion_resistance': 7,
                'max_cost': 15.0,
                'material_types': ['Steel', 'Aluminum', 'Composite']
            },
            'construction': {
                'min_strength': 100,
                'max_cost': 2.0,
                'min_availability': 9,
                'material_types': ['Steel', 'Concrete', 'Aluminum']
            }
        }
        
        if application_type not in application_constraints:
            available_types = list(application_constraints.keys())
            raise ValueError(f"Unknown application type. Available types: {available_types}")
        
        constraints = application_constraints[application_type]
        results = self.optimize_material_selection(constraints)
        
        results['application_type'] = application_type
        results['application_constraints'] = constraints
        
        return results
    
    def compare_materials(self, material_ids: List[int]) -> pd.DataFrame:
        """
        Compare specific materials side by side.
        
        Args:
            material_ids (List): List of material IDs to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_materials = self.materials_data[
            self.materials_data['material_id'].isin(material_ids)
        ].copy()
        
        if len(comparison_materials) == 0:
            raise ValueError("No materials found with the provided IDs")
        
        # Calculate normalized scores for comparison
        comparison_materials = self.calculate_material_scores(comparison_materials)
        
        # Select key columns for comparison
        comparison_columns = [
            'material_name', 'material_type', 'tensile_strength_mpa', 
            'cost_per_kg_usd', 'density_kg_m3', 'availability_score',
            'environmental_impact_score', 'corrosion_resistance_score',
            'composite_score'
        ]
        
        return comparison_materials[comparison_columns].round(3)

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from data_processor import MaterialDataProcessor
    
    # Load and process data
    processor = MaterialDataProcessor()
    data_path = "../data/materials.csv"
    
    if os.path.exists(data_path):
        processor.load_data(data_path)
        materials_data = processor.data
        
        # Initialize optimizer
        optimizer = MaterialOptimizer(materials_data)
        
        # Example optimization
        constraints = {
            'max_cost': 10.0,
            'min_strength': 400,
            'min_availability': 6
        }
        
        results = optimizer.optimize_material_selection(constraints)
        print(f"Optimization status: {results['status']}")
        if results['top_material']:
            print(f"Top recommended material: {results['top_material']['name']}")
    else:
        print(f"Data file not found: {data_path}")
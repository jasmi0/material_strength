import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_processor import MaterialDataProcessor
    from src.models import MaterialModelManager
    from src.optimization import MaterialOptimizer
except ImportError:
    from data_processor import MaterialDataProcessor
    from models import MaterialModelManager
    from optimization import MaterialOptimizer

@st.cache_data
def load_data():
    try:
        data_path = "data/materials.csv"
        if not os.path.exists(data_path):
            data_path = "../data/materials.csv"
        
        processor = MaterialDataProcessor()
        processor.load_data(data_path)
        return processor.data, processor
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_resource
def load_models():
    try:
        model_manager = MaterialModelManager()
        model_dir = "models"
        if not os.path.exists(model_dir):
            model_dir = "../models"
        
        if os.path.exists(os.path.join(model_dir, "strength_model.pkl")):
            model_manager.load_models(model_dir)
            return model_manager
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load pre-trained models: {str(e)}")
        return None

def get_application_descriptions():
    return {
        'structural': "General structural applications requiring high strength and moderate cost",
        'aerospace': "Aerospace applications requiring high strength-to-weight ratio and reliability",
        'automotive': "Automotive applications balancing strength, weight, and cost",
        'marine': "Marine applications requiring corrosion resistance and durability",
        'construction': "Construction applications emphasizing cost-effectiveness and availability"
    }

def get_expected_columns():
    return [
        'material_id', 'material_name', 'material_type', 'tensile_strength_mpa',
        'yield_strength_mpa', 'density_kg_m3', 'cost_per_kg_usd', 'availability_score',
        'environmental_impact_score', 'elastic_modulus_gpa', 'hardness_hv', 'corrosion_resistance_score'
    ]

def validate_uploaded_data(uploaded_data, expected_columns):
    missing_cols = [col for col in expected_columns if col not in uploaded_data.columns]
    return missing_cols

def get_display_columns():
    return [
        'material_name', 'material_type', 'tensile_strength_mpa', 
        'cost_per_kg_usd', 'availability_score', 'environmental_impact_score',
        'composite_score'
    ]

def apply_data_filters(materials_data, material_types, strength_range, cost_range, availability_min):
    return materials_data[
        (materials_data['material_type'].isin(material_types)) &
        (materials_data['tensile_strength_mpa'].between(strength_range[0], strength_range[1])) &
        (materials_data['cost_per_kg_usd'].between(cost_range[0], cost_range[1])) &
        (materials_data['availability_score'] >= availability_min)
    ]

def build_constraints(max_cost, min_strength, min_availability, min_environmental, 
                     max_density, min_corrosion, material_types):
    return {
        'max_cost': max_cost,
        'min_strength': min_strength,
        'min_availability': min_availability,
        'min_environmental': min_environmental,
        'max_density': max_density,
        'min_corrosion_resistance': min_corrosion,
        'material_types': material_types if material_types else None
    }

def build_weights(w_strength, w_cost, w_availability, w_environmental, w_corrosion):
    return {
        'strength': w_strength,
        'cost': w_cost,
        'availability': w_availability,
        'environmental': w_environmental,
        'corrosion_resistance': w_corrosion
    }
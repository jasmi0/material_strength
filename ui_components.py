import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

def create_material_comparison_chart(materials_df: pd.DataFrame, x_axis: str, y_axis: str, color_by: str):
    fig = px.scatter(
        materials_df,
        x=x_axis,
        y=y_axis,
        color=color_by,
        hover_data=['material_name', 'material_type'],
        title=f"{y_axis} vs {x_axis} colored by {color_by}",
        labels={
            x_axis: x_axis.replace('_', ' ').title(),
            y_axis: y_axis.replace('_', ' ').title(),
            color_by: color_by.replace('_', ' ').title()
        }
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def create_material_properties_radar(material_data: pd.Series):
    properties = {
        'Strength': material_data['tensile_strength_mpa'] / 50,
        'Cost Efficiency': 100 - (material_data['cost_per_kg_usd'] / 2),
        'Availability': material_data['availability_score'] * 10,
        'Environmental': material_data['environmental_impact_score'] * 10,
        'Corrosion Resistance': material_data['corrosion_resistance_score'] * 10,
        'Density Efficiency': 100 - (material_data['density_kg_m3'] / 100)
    }
    
    for key, value in properties.items():
        properties[key] = max(0, min(100, value))
    
    categories = list(properties.keys())
    values = list(properties.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=material_data['material_name']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title=f"Properties Overview: {material_data['material_name']}"
    )
    
    return fig

def create_optimization_results_chart(results: Dict[str, Any]):
    if results['status'] != 'success':
        return None
    
    top_materials = results['recommended_materials'].head(10)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Composite Scores', 'Strength vs Cost', 'Material Types', 'Availability vs Environmental Impact'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=top_materials['material_name'],
            y=top_materials['composite_score'],
            name='Composite Score',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=top_materials['cost_per_kg_usd'],
            y=top_materials['tensile_strength_mpa'],
            mode='markers',
            text=top_materials['material_name'],
            marker=dict(
                size=top_materials['composite_score']*20,
                color=top_materials['composite_score'],
                colorscale='viridis'
            ),
            name='Materials'
        ),
        row=1, col=2
    )
    
    type_counts = top_materials['material_type'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            name='Material Types'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=top_materials['availability_score'],
            y=top_materials['environmental_impact_score'],
            mode='markers',
            text=top_materials['material_name'],
            marker=dict(
                size=10,
                color=top_materials['composite_score'],
                colorscale='plasma'
            ),
            name='Sustainability'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def create_multi_material_radar(comparison_df: pd.DataFrame):
    fig = go.Figure()
    
    for _, material in comparison_df.head(3).iterrows():
        properties = {
            'Strength': material['tensile_strength_mpa'] / 50,
            'Cost Efficiency': 100 - (material['cost_per_kg_usd'] / 2),
            'Availability': material['availability_score'] * 10,
            'Environmental': material['environmental_impact_score'] * 10,
            'Corrosion Resistance': material['corrosion_resistance_score'] * 10,
            'Density Efficiency': 100 - (material['density_kg_m3'] / 100)
        }
        
        values = [max(0, min(100, v)) for v in properties.values()]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=list(properties.keys()),
            fill='toself',
            name=material['material_name']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Material Properties Comparison"
    )
    
    return fig

def get_app_styles():
    return """
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
            color: #1f77b4;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #ff7f0e;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .success-message {
            color: #28a745;
            font-weight: bold;
        }
        .warning-message {
            color: #ffc107;
            font-weight: bold;
        }
        .error-message {
            color: #dc3545;
            font-weight: bold;
        }
    </style>
    """
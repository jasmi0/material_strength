import streamlit as st
import pandas as pd
import plotly.express as px
from app_utils import (
    load_data, load_models, get_application_descriptions, get_expected_columns,
    validate_uploaded_data, get_display_columns, apply_data_filters,
    build_constraints, build_weights
)
from ui_components import (
    create_material_comparison_chart, create_material_properties_radar,
    create_optimization_results_chart, create_multi_material_radar, get_app_styles
)
from src.optimization import MaterialOptimizer

st.set_page_config(
    page_title="Material Strength & Selection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(get_app_styles(), unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">Material Strength & Selection System</div>', unsafe_allow_html=True)
    st.markdown("**AI-powered material selection for engineering applications**")
    
    materials_data, processor = load_data()
    if materials_data is None:
        st.error("Failed to load material data. Please check if the data file exists.")
        return
    
    model_manager = load_models()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Material Explorer", "Optimization Tool", "Application Recommendations", "Material Comparison", "Data Upload"]
    )
    
    optimizer = MaterialOptimizer(materials_data, model_manager)
    
    if page == "Material Explorer":
        st.markdown('<div class="section-header">Material Database Explorer</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Materials", len(materials_data))
        with col2:
            st.metric("Material Types", materials_data['material_type'].nunique())
        with col3:
            st.metric("Avg Strength (MPa)", f"{materials_data['tensile_strength_mpa'].mean():.0f}")
        with col4:
            st.metric("Avg Cost ($/kg)", f"{materials_data['cost_per_kg_usd'].mean():.2f}")
        
        st.subheader("Filters")
        col1, col2 = st.columns(2)
        
        with col1:
            material_types = st.multiselect(
                "Material Types",
                options=materials_data['material_type'].unique(),
                default=materials_data['material_type'].unique()
            )
            
            strength_range = st.slider(
                "Tensile Strength Range (MPa)",
                min_value=int(materials_data['tensile_strength_mpa'].min()),
                max_value=int(materials_data['tensile_strength_mpa'].max()),
                value=(int(materials_data['tensile_strength_mpa'].min()), int(materials_data['tensile_strength_mpa'].max()))
            )
        
        with col2:
            cost_range = st.slider(
                "Cost Range ($/kg)",
                min_value=float(materials_data['cost_per_kg_usd'].min()),
                max_value=float(materials_data['cost_per_kg_usd'].max()),
                value=(float(materials_data['cost_per_kg_usd'].min()), float(materials_data['cost_per_kg_usd'].max()))
            )
            
            availability_min = st.slider(
                "Minimum Availability Score",
                min_value=0,
                max_value=10,
                value=0
            )
        
        filtered_data = apply_data_filters(materials_data, material_types, strength_range, cost_range, availability_min)
        
        st.subheader(f"Filtered Results ({len(filtered_data)} materials)")
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis", options=['tensile_strength_mpa', 'cost_per_kg_usd', 'density_kg_m3', 'availability_score'])
        with col2:
            y_axis = st.selectbox("Y-axis", options=['cost_per_kg_usd', 'tensile_strength_mpa', 'environmental_impact_score', 'corrosion_resistance_score'])
        
        color_by = st.selectbox("Color by", options=['material_type', 'availability_score', 'environmental_impact_score'])
        
        if len(filtered_data) > 0:
            chart = create_material_comparison_chart(filtered_data, x_axis, y_axis, color_by)
            st.plotly_chart(chart, use_container_width=True)
            
            st.dataframe(filtered_data[['material_name', 'material_type', 'tensile_strength_mpa', 'cost_per_kg_usd', 'availability_score']])
        else:
            st.warning("No materials match the selected filters.")
    
    elif page == "Optimization Tool":
        st.markdown('<div class="section-header">Material Optimization Tool</div>', unsafe_allow_html=True)
        
        st.subheader("Set Your Constraints")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_cost = st.number_input("Maximum Cost ($/kg)", min_value=0.0, value=50.0, step=0.1)
            min_strength = st.number_input("Minimum Tensile Strength (MPa)", min_value=0, value=200, step=10)
            min_availability = st.slider("Minimum Availability Score", 0, 10, 5)
            min_environmental = st.slider("Minimum Environmental Impact Score", 0, 10, 5)
        
        with col2:
            max_density = st.number_input("Maximum Density (kg/m³)", min_value=0, value=10000, step=100)
            min_corrosion = st.slider("Minimum Corrosion Resistance", 0, 10, 0)
            material_types = st.multiselect(
                "Allowed Material Types (leave empty for all)",
                options=materials_data['material_type'].unique()
            )
        
        st.subheader("Optimization Weights")
        st.write("Adjust the importance of different criteria (must sum to 1.0)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            w_strength = st.slider("Strength", 0.0, 1.0, 0.3, 0.05)
        with col2:
            w_cost = st.slider("Cost", 0.0, 1.0, 0.25, 0.05)
        with col3:
            w_availability = st.slider("Availability", 0.0, 1.0, 0.2, 0.05)
        with col4:
            w_environmental = st.slider("Environmental", 0.0, 1.0, 0.15, 0.05)
        with col5:
            w_corrosion = st.slider("Corrosion Resistance", 0.0, 1.0, 0.1, 0.05)
        
        weights = build_weights(w_strength, w_cost, w_availability, w_environmental, w_corrosion)
        
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            st.warning(f"Weights sum to {weight_sum:.2f}. Please adjust to sum to 1.0")
        
        if st.button("Optimize Material Selection", type="primary"):
            constraints = build_constraints(max_cost, min_strength, min_availability, min_environmental, 
                                          max_density, min_corrosion, material_types)
            
            try:
                results = optimizer.optimize_material_selection(constraints, weights, top_n=15)
                
                if results['status'] == 'success':
                    st.success(f"Optimization successful! Found {results['feasible_count']} feasible materials.")
                    
                    top_material = results['top_material']
                    st.subheader("Top Recommendation")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Material", top_material['name'])
                    with col2:
                        st.metric("Type", top_material['type'])
                    with col3:
                        st.metric("Score", f"{top_material['score']:.3f}")
                    with col4:
                        st.metric("Strength (MPa)", f"{top_material['strength']:.0f}")
                    
                    chart = create_optimization_results_chart(results)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    
                    st.subheader("Detailed Results")
                    recommended_materials = results['recommended_materials']
                    display_columns = get_display_columns()
                    st.dataframe(
                        recommended_materials[display_columns].round(3),
                        use_container_width=True
                    )
                    
                    if len(recommended_materials) > 0:
                        st.subheader("Top Material Properties")
                        radar_chart = create_material_properties_radar(recommended_materials.iloc[0])
                        st.plotly_chart(radar_chart, use_container_width=True)
                
                else:
                    st.error("No materials satisfy the given constraints. Please relax your requirements.")
                    
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
    
    elif page == "Application Recommendations":
        st.markdown('<div class="section-header">Application-Specific Recommendations</div>', unsafe_allow_html=True)
        
        application_type = st.selectbox(
            "Select Engineering Application",
            options=['structural', 'aerospace', 'automotive', 'marine', 'construction']
        )
        
        application_descriptions = get_application_descriptions()
        
        st.info(f"**{application_type.title()} Applications:** {application_descriptions[application_type]}")
        
        if st.button(f"Get {application_type.title()} Recommendations", type="primary"):
            try:
                results = optimizer.get_material_recommendations(application_type)
                
                if results['status'] == 'success':
                    st.success(f"Found {results['feasible_count']} suitable materials for {application_type} applications")
                    
                    st.subheader("Application Constraints")
                    constraints_df = pd.DataFrame([results['application_constraints']]).T
                    constraints_df.columns = ['Value']
                    st.dataframe(constraints_df)
                    
                    st.subheader("Top Recommendations")
                    recommended_materials = results['recommended_materials'].head(10)
                    
                    for i, (_, material) in enumerate(recommended_materials.iterrows()):
                        with st.expander(f"{i+1}. {material['material_name']} (Score: {material['composite_score']:.3f})"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Type:** {material['material_type']}")
                                st.write(f"**Strength:** {material['tensile_strength_mpa']:.0f} MPa")
                                st.write(f"**Cost:** ${material['cost_per_kg_usd']:.2f}/kg")
                            with col2:
                                st.write(f"**Density:** {material['density_kg_m3']:.0f} kg/m³")
                                st.write(f"**Availability:** {material['availability_score']}/10")
                                st.write(f"**Environmental:** {material['environmental_impact_score']}/10")
                            with col3:
                                st.write(f"**Corrosion Resistance:** {material['corrosion_resistance_score']}/10")
                                st.write(f"**Yield Strength:** {material['yield_strength_mpa']:.0f} MPa")
                                st.write(f"**Elastic Modulus:** {material['elastic_modulus_gpa']:.0f} GPa")
                
                else:
                    st.error("No suitable materials found for this application.")
                    
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")
    
    elif page == "Material Comparison":
        st.markdown('<div class="section-header">Material Comparison Tool</div>', unsafe_allow_html=True)
        
        st.subheader("Select Materials to Compare")
        
        material_names = materials_data['material_name'].tolist()
        selected_materials = st.multiselect(
            "Choose materials (max 5)",
            options=material_names,
            max_selections=5
        )
        
        if selected_materials:
            material_ids = materials_data[materials_data['material_name'].isin(selected_materials)]['material_id'].tolist()
            
            try:
                comparison_df = optimizer.compare_materials(material_ids)
                
                st.subheader("Comparison Table")
                st.dataframe(comparison_df, use_container_width=True)
                
                if len(comparison_df) > 1:
                    st.subheader("Visual Comparison")
                    
                    fig1 = px.scatter(
                        comparison_df,
                        x='cost_per_kg_usd',
                        y='tensile_strength_mpa',
                        size='composite_score',
                        color='material_type',
                        hover_data=['material_name'],
                        title="Strength vs Cost Comparison"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    fig2 = create_multi_material_radar(comparison_df)
                    st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating comparison: {str(e)}")
        else:
            st.info("Please select materials to compare.")
    
    elif page == "Data Upload":
        st.markdown('<div class="section-header">Upload Your Own Material Data</div>', unsafe_allow_html=True)
        
        st.write("Upload a CSV file with your own material data to use with the optimization tools.")
        
        st.subheader("Expected CSV Format")
        expected_columns = get_expected_columns()
        st.write("Required columns:")
        for col in expected_columns:
            st.write(f"- {col}")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                
                st.success("File uploaded successfully!")
                st.subheader("Data Preview")
                st.dataframe(uploaded_data.head())
                
                missing_cols = validate_uploaded_data(uploaded_data, expected_columns)
                if missing_cols:
                    st.warning(f"Missing columns: {missing_cols}")
                else:
                    st.success("All required columns present!")
                    
                    if st.button("Use This Data for Analysis"):
                        st.session_state['custom_data'] = uploaded_data
                        st.success("Custom data loaded! You can now use it in other tools.")
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    st.markdown("---")
    st.markdown("**Material Strength & Selection System** - AI-powered material selection for engineering applications")

if __name__ == "__main__":
    main()
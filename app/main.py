import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar(data):
    st.sidebar.header('Cell Nuclei Measurements')

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}
    
    for label, key in slider_labels:
        min_value = float(data[key].min())
        max_value = float(data[key].max())
        default_value = float(data[key].mean())
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default_value
        )
    
    return input_dict

def get_scaled_values(input_dict):
    # Define a Function to Scale the Values based on the Min and Max of the Predictor in the Training Data
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(inputs):
    inputs = get_scaled_values(inputs)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            inputs['radius_mean'], inputs['texture_mean'], inputs['perimeter_mean'],
            inputs['area_mean'], inputs['smoothness_mean'], inputs['compactness_mean'],
            inputs['concavity_mean'], inputs['concave points_mean'], inputs['symmetry_mean'],
            inputs['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            inputs['radius_se'], inputs['texture_se'], inputs['perimeter_se'], inputs['area_se'],
            inputs['smoothness_se'], inputs['compactness_se'], inputs['concavity_se'],
            inputs['concave points_se'], inputs['symmetry_se'], inputs['fractal_dimension_se']
            ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            inputs['radius_worst'], inputs['texture_worst'], inputs['perimeter_worst'],
            inputs['area_worst'], inputs['smoothness_worst'], inputs['compactness_worst'],
            inputs['concavity_worst'], inputs['concave points_worst'], inputs['symmetry_worst'],
            inputs['fractal_dimension_worst']
            ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def add_predictions(inputs):
    model = pickle.load(open('model/model.pkl','rb'))
    scaler = pickle.load(open('model/scaler.pkl','rb'))
    input_array = np.array(list(inputs.values())).reshape(1,-1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write('Benign')
    else:
        st.write('Malicious')

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def main():
    st.set_page_config(
        page_title='Breast Cancer Predictor',
        page_icon=':female-doctor:',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    data = get_clean_data()

    # Add sidebar and get user inputs
    inputs = add_sidebar(data)

    with st.container():
        st.title('Breast Cancer Predictor')
        st.write(
            "Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. "
            "This app predicts whether a breast mass is benign or malignant based on the measurements it receives from your cytology lab. "
            "You can also update the measurements by hand using the sliders in the sidebar."
        )
    
    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(inputs)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(inputs)

if __name__ == '__main__':
    main()

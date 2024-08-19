from joblib import load
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the trained model (update the path as per your directory structure)
model = load('final_random_forest_model.joblib')

# Define the expected feature columns
feature_columns = ['HST', 'AST', 'HTGDIFF', 'ATGDIFF', 'AVGATGDIFF', 'AVGFTAG', 'AVGFTHG', 'AVGHTGDIFF',
                   'HTWinStreak3', 'HTLossStreak3', 'ATWinStreak3', 'ATLossStreak3', 'HTWinStreak5', 
                   'HTLossStreak5', 'ATWinStreak5', 'ATLossStreak5']

# Streamlit app layout
st.set_page_config(page_title="Football Match Outcome Prediction", layout="centered", page_icon="⚽")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
        }

        .main {
            background: linear-gradient(135deg, #1f4037, #99f2c8);
            color: white;
        }

        .block-container {
            background: rgba(0, 0, 0, 0.8);
            padding: 2rem;
            border-radius: 1rem;
        }

        h1 {
            color: #e0e0e0;
            font-weight: 700;
        }

        label {
            color: white !important;
        }

        .stButton button {
            background: linear-gradient(to right, #ff416c, #ff4b2b);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 1rem;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: background 0.3s;
        }

        .stButton button:hover {
            background: linear-gradient(to right, #ff4b2b, #ff416c);
        }

        .css-1aumxhk {
            background: rgba(255, 255, 255, 0.8) !important;
        }

        .header-container {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header-title {
            font-size: 2rem;
            font-weight: bold;
            margin: 0;
        }

        .header-subtitle {
            font-size: 1.2rem;
            margin: 0.5rem 0;
        }

        .header-uni-logo {
            width: 150px;
            height: 150px;
            display: block;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# Display university logo in center
st.markdown('<div class="header-container">'
            '<div class="header-title">Technological University of Mawlamyine</div>'
            '<div class="header-subtitle">Department of Information Technology</div>'
            '<div class="header-subtitle">Supervisor: Tr. Daw Nyein Nyein Thu | Student: Sai Pyae Sone Thu</div>'
            '</div>', unsafe_allow_html=True)

st.title('Football Match Outcome Prediction')

# Create a form for user inputs
with st.form("prediction_form"):
    st.write("Enter the details of the match:")
    
    # Create input fields for each feature
    hst = st.number_input('Home Team Shots on Target (HST)', min_value=0, step=1)
    ast = st.number_input('Away Team Shots on Target (AST)', min_value=0, step=1)
    htgdiff = st.number_input('Home Team Goal Difference (HTGDIFF)', min_value=0, step=1)
    atgdiff = st.number_input('Away Team Goal Difference (ATGDIFF)', min_value=0, step=1)
    avgatgdiff = st.number_input('Average Away Team Goal Difference (AVGATGDIFF)', min_value=0.0, step=0.1)
    avgftag = st.number_input('Average Full Time Away Goals (AVGFTAG)', min_value=0.0, step=0.1)
    avgfthg = st.number_input('Average Full Time Home Goals (AVGFTHG)', min_value=0.0, step=0.1)
    avghtgdiff = st.number_input('Average Home Team Goal Difference (AVGHTGDIFF)', min_value=0.0, step=0.1)
    htwin_streak3 = st.number_input('Home Team Win Streak in Last 3 Matches (HTWinStreak3)', min_value=0, step=1)
    htloss_streak3 = st.number_input('Home Team Loss Streak in Last 3 Matches (HTLossStreak3)', min_value=0, step=1)
    atwin_streak3 = st.number_input('Away Team Win Streak in Last 3 Matches (ATWinStreak3)', min_value=0, step=1)
    atloss_streak3 = st.number_input('Away Team Loss Streak in Last 3 Matches (ATLossStreak3)', min_value=0, step=1)
    htwin_streak5 = st.number_input('Home Team Win Streak in Last 5 Matches (HTWinStreak5)', min_value=0, step=1)
    htloss_streak5 = st.number_input('Home Team Loss Streak in Last 5 Matches (HTLossStreak5)', min_value=0, step=1)
    atwin_streak5 = st.number_input('Away Team Win Streak in Last 5 Matches (ATWinStreak5)', min_value=0, step=1)
    atloss_streak5 = st.number_input('Away Team Loss Streak in Last 5 Matches (ATLossStreak5)', min_value=0, step=1)

    # Submit button
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Collect input data into a DataFrame
        input_data = pd.DataFrame({
            'HST': [hst],
            'AST': [ast],
            'HTGDIFF': [htgdiff],
            'ATGDIFF': [atgdiff],
            'AVGATGDIFF': [avgatgdiff],
            'AVGFTAG': [avgftag],
            'AVGFTHG': [avgfthg],
            'AVGHTGDIFF': [avghtgdiff],
            'HTWinStreak3': [htwin_streak3],
            'HTLossStreak3': [htloss_streak3],
            'ATWinStreak3': [atwin_streak3],
            'ATLossStreak3': [atloss_streak3],
            'HTWinStreak5': [htwin_streak5],
            'HTLossStreak5': [htloss_streak5],
            'ATWinStreak5': [atwin_streak5],
            'ATLossStreak5': [atloss_streak5]
        })

        # Make predictions
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        # Show predictions and probabilities
        st.write("Prediction:")
        outcome_map = {1: 'Home Win', 0: 'Draw', -1: 'Away Win'}
        st.write(f"Predicted Outcome: {outcome_map[prediction[0]]}")
        
        st.write("Predicted Probabilities:")
        labels = ['Away Win', 'Draw', 'Home Win']
        prob_dict = {label: prob for label, prob in zip(labels, probabilities[0])}
        st.write(prob_dict)

        # Visualize probabilities using Plotly
        fig = go.Figure(data=[go.Bar(x=labels, y=probabilities[0], marker_color=['#e74c3c', '#f39c12', '#27ae60'])])
        fig.update_layout(title='Predicted Probabilities', xaxis_title='Outcome', yaxis_title='Probability', template='plotly_dark')
        st.plotly_chart(fig)

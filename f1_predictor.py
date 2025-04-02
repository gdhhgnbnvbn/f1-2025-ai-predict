import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import joblib
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data_loader import F1DataLoader

# Calendario F1 2025
F1_CALENDAR_2025 = [
    "Australian Grand Prix - Melbourne (16 Mar) - COMPLETED",
    "Chinese Grand Prix - Shanghai (23 Mar) - COMPLETED",
    "Japanese Grand Prix - Suzuka (6 Apr)",
    "Bahrain Grand Prix - Sakhir (13 Apr)",
    "Saudi Arabian Grand Prix - Jeddah (20 Apr)",
    "Miami Grand Prix - Miami (4 May)",
    "Emilia Romagna Grand Prix - Imola (18 May)",
    "Monaco Grand Prix - Monte Carlo (25 May)",
    "Spanish Grand Prix - Barcelona (1 Jun)",
    "Canadian Grand Prix - Montreal (15 Jun)",
    "Austrian Grand Prix - Spielberg (29 Jun)",
    "British Grand Prix - Silverstone (6 Jul)",
    "Belgian Grand Prix - Spa-Francorchamps (27 Jul)",
    "Hungarian Grand Prix - Budapest (3 Aug)",
    "Dutch Grand Prix - Zandvoort (31 Aug)",
    "Italian Grand Prix - Monza (7 Sep)",
    "Azerbaijan Grand Prix - Baku (21 Sep)",
    "Singapore Grand Prix - Singapore (5 Oct)",
    "United States Grand Prix - Austin (19 Oct)",
    "Mexico City Grand Prix - Mexico City (26 Oct)",
    "São Paulo Grand Prix - São Paulo (9 Nov)",
    "Las Vegas Grand Prix - Las Vegas (22 Nov)",
    "Qatar Grand Prix - Lusail (30 Nov)",
    "Abu Dhabi Grand Prix - Yas Marina (7 Dec)"
]

class F1Predictor:
    def __init__(self, data_path='f1data'):
        self.data_path = data_path
        self.model = None
        self.data_loader = F1DataLoader(data_path)
        self.feature_importance = None
        self.grid_2025 = None
        self.results_2025 = None
        self.load_2025_data()
        # Try to load existing model at initialization
        try:
            self.load_model()
        except:
            pass
    
    def load_2025_data(self):
        """Load the 2025 F1 grid and results data."""
        try:
            self.grid_2025 = pd.read_csv(f'{self.data_path}/f1_2025_grid.csv')
            self.results_2025 = pd.read_csv(f'{self.data_path}/f1_2025_results.csv')
        except Exception as e:
            print(f"Error loading 2025 data: {e}")
    
    def get_driver_recent_results(self, driver_name):
        """Get recent results for a driver in 2025."""
        if self.results_2025 is None:
            return None
        
        driver_results = self.results_2025[self.results_2025['driver_name'] == driver_name]
        return driver_results.sort_values('date', ascending=False)
    
    def predict_2025_race(self, circuit_name, qualifying_results=None):
        """
        Predict the outcome of a 2025 race.
        
        Args:
            circuit_name (str): Name of the circuit
            qualifying_results (dict): Optional dictionary with grid positions for each driver
        """
        if self.model is None or self.grid_2025 is None:
            return None
            
        # Create a prediction dataframe
        pred_df = self.grid_2025.copy()
        
        # Add default values for required features
        pred_df['grid'] = range(1, len(pred_df) + 1)  # Default grid positions
        if qualifying_results:
            for driver_id, position in qualifying_results.items():
                pred_df.loc[pred_df['driverId'] == driver_id, 'grid'] = position
        
        # Add other required features with reasonable default values
        pred_df['qual_position_avg'] = pred_df['grid']
        
        # Update points_moving_avg based on 2025 results
        pred_df['points_moving_avg'] = 0
        if self.results_2025 is not None:
            for idx, row in pred_df.iterrows():
                recent_results = self.get_driver_recent_results(row['driver_name'])
                if recent_results is not None and not recent_results.empty:
                    # Calculate points based on positions (simplified)
                    points = recent_results['position'].map(lambda x: max(26-x, 0)).mean()
                    pred_df.loc[idx, 'points_moving_avg'] = points
        
        pred_df['circuit_wins'] = 0  # Could be updated with historical data
        pred_df['points_championship'] = pred_df['points_moving_avg']
        
        # Calculate championship positions based on points
        pred_df['position_championship'] = pred_df['points_championship'].rank(ascending=False, method='min')
        
        # Calculate constructor stats
        constructor_stats = pred_df.groupby('team_name').agg({
            'points_moving_avg': ['mean', 'std']
        }).reset_index()
        constructor_stats.columns = ['team_name', 'constructor_points_mean', 'constructor_points_std']
        
        pred_df = pd.merge(pred_df, constructor_stats, on='team_name', how='left')
        pred_df['constructor_position_mean'] = pred_df['constructor_points_mean'].rank(ascending=False, method='min')
        
        # Encode categorical variables
        for col, encoder in self.data_loader.label_encoders.items():
            if col == 'nationality':
                pred_df[f'{col}_encoded'] = encoder.transform(pred_df['nationality'])
            elif col == 'nationality_constructor':
                pred_df[f'{col}_encoded'] = encoder.transform(pred_df['constructor_nationality'])
            elif col == 'country':
                # Use a default value for now
                pred_df[f'{col}_encoded'] = 0
        
        # Select features in the same order as training
        feature_columns = [
            'grid',
            'qual_position_avg',
            'points_moving_avg',
            'circuit_wins',
            'points_championship',
            'position_championship',
            'constructor_points_mean',
            'constructor_points_std',
            'constructor_position_mean',
            'nationality_encoded',
            'nationality_constructor_encoded',
            'country_encoded'
        ]
        
        # Get win probabilities for each driver
        win_probs = self.model.predict_proba(pred_df[feature_columns])[:, 1]
        
        # Create results dataframe
        results = pd.DataFrame({
            'Driver': pred_df['driver_name'],
            'Team': pred_df['team_name'],
            'Grid': pred_df['grid'],
            'Win Probability': win_probs,
            'Championship Points': pred_df['points_championship']
        })
        
        return results.sort_values('Win Probability', ascending=False).reset_index(drop=True)
    
    def simulate_championship(self):
        """Simulate the remaining races of the 2025 championship."""
        if self.model is None or self.grid_2025 is None:
            return None
            
        # Initialize championship points with actual results from 2025
        championship_points = {driver: 0 for driver in self.grid_2025['driver_name']}
        
        # Add points from actual races
        if self.results_2025 is not None:
            for _, race in self.results_2025.iterrows():
                # F1 points system
                points = {
                    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
                }
                if race['position'] in points:
                    championship_points[race['driver_name']] += points[race['position']]
                # Add point for fastest lap if applicable
                if pd.notna(race['fastest_lap']):
                    championship_points[race['driver_name']] += 1
        
        # Get remaining races
        completed_races = set()
        if self.results_2025 is not None:
            completed_races = set(self.results_2025['race_name'].unique())
        
        remaining_races = [
            race.split(" (")[0] for race in F1_CALENDAR_2025 
            if not race.endswith("COMPLETED")
        ]
        
        # Team reliability factors (1.0 = perfect reliability, higher = more problems)
        team_reliability = {
            'Red Bull Racing': 1.05,  # Top team with best reliability
            'Ferrari': 1.2,
            'Mercedes': 1.15,
            'McLaren': 1.2,
            'Aston Martin': 1.25,
            'Alpine': 1.3,
            'Williams': 1.35,
            'RB': 1.3,  # Ex AlphaTauri
            'Kick Sauber': 1.35,
            'Haas F1 Team': 1.4
        }
        
        # Driver error probability factors (higher = more prone to errors)
        driver_error_factor = {
            'Max Verstappen': 0.04,  # Campione in carica
            'Yuki Tsunoda': 0.08,    # Promosso in Red Bull
            'Charles Leclerc': 0.06,
            'Lewis Hamilton': 0.05,   # Esperienza in Ferrari
            'George Russell': 0.07,
            'Andrea Kimi Antonelli': 0.12,  # Rookie in Mercedes
            'Lando Norris': 0.06,
            'Oscar Piastri': 0.07,
            'Fernando Alonso': 0.05,  # Esperienza
            'Lance Stroll': 0.09,
            'Pierre Gasly': 0.08,
            'Jack Doohan': 0.11,      # Rookie
            'Alexander Albon': 0.08,
            'Carlos Sainz': 0.07,     # In Williams
            'Esteban Ocon': 0.08,
            'Oliver Bearman': 0.1,    # Rookie in Haas
            'Nico Hulkenberg': 0.08,
            'Gabriel Bortoleto': 0.12, # Rookie
            'Liam Lawson': 0.09,      # In RB
            'Isack Hadjar': 0.11      # Rookie in RB
        }
        
        # Simulate each remaining race
        race_results = []
        for race in remaining_races:
            # Predict race outcome
            results = self.predict_2025_race(race)
            if results is None:
                continue
            
            # Convert results to list for manipulation
            race_order = results.to_dict('records')
            
            # Simulate race incidents
            for i, driver in enumerate(race_order):
                team = driver['Team']
                driver_name = driver['Driver']
                
                # 1. DNF probability based on team reliability and track position
                base_dnf_prob = 0.02  # 2% base probability
                position_factor = 1 + (i * 0.01)  # Slightly higher chance of DNF for cars further back
                team_factor = team_reliability.get(team, 1.2)
                dnf_probability = base_dnf_prob * position_factor * team_factor
                
                if np.random.random() < dnf_probability:
                    race_order[i]['DNF'] = True
                    continue
                
                # 2. Driver errors (spins, missed braking points, etc.)
                error_prob = driver_error_factor.get(driver_name, 0.1)
                if np.random.random() < error_prob:
                    # Lose 2-5 positions
                    positions_lost = np.random.randint(2, 6)
                    new_pos = min(len(race_order) - 1, i + positions_lost)
                    # Swap positions
                    race_order.insert(new_pos, race_order.pop(i))
                
                # 3. Random penalties (5 or 10 seconds)
                if np.random.random() < 0.05:  # 5% chance of penalty
                    penalty_time = np.random.choice([5, 10])
                    # Simulate penalty effect on position
                    positions_lost = np.random.randint(1, 4)
                    new_pos = min(len(race_order) - 1, i + positions_lost)
                    race_order.insert(new_pos, race_order.pop(i))
                
                # 4. Pit stop issues
                if np.random.random() < 0.08:  # 8% chance of slow pit stop
                    # Lose 1-3 positions
                    positions_lost = np.random.randint(1, 4)
                    new_pos = min(len(race_order) - 1, i + positions_lost)
                    race_order.insert(new_pos, race_order.pop(i))
            
            # 5. Safety Car periods (30% chance per race)
            if np.random.random() < 0.3:
                # Safety Car bunches up the field and can lead to position swaps
                # Randomly swap some positions in the top 10
                for _ in range(np.random.randint(1, 4)):
                    pos1, pos2 = np.random.randint(0, min(10, len(race_order)), size=2)
                    race_order[pos1], race_order[pos2] = race_order[pos2], race_order[pos1]
            
            # Calculate points and update championship
            for pos, driver in enumerate(race_order):
                if driver.get('DNF', False):
                    continue
                    
                points = 0
                if pos < 10:  # Points positions
                    points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                    points = points_system[pos]
                    
                    # Fastest lap point (more likely for top 5, but possible for others in top 10)
                    fastest_lap_prob = 0.3 if pos < 5 else 0.1
                    if np.random.random() < fastest_lap_prob:
                        points += 1
                
                championship_points[driver['Driver']] += points
            
            # Store race results (only non-DNF drivers)
            valid_results = [d for d in race_order if not d.get('DNF', False)]
            if len(valid_results) >= 3:
                race_results.append({
                    'Race': race,
                    'Winner': valid_results[0]['Driver'],
                    'Second': valid_results[1]['Driver'],
                    'Third': valid_results[2]['Driver']
                })
        
        # Create final championship standings
        final_standings = pd.DataFrame({
            'Driver': list(championship_points.keys()),
            'Points': list(championship_points.values())
        })
        
        # Add team information and sort by points
        final_standings = pd.merge(
            final_standings,
            self.grid_2025[['driver_name', 'team_name']],
            left_on='Driver',
            right_on='driver_name'
        ).drop('driver_name', axis=1)
        
        final_standings = final_standings.sort_values('Points', ascending=False).reset_index(drop=True)
        
        # Calculate constructor standings
        constructor_standings = final_standings.groupby('team_name')['Points'].sum().reset_index()
        constructor_standings = constructor_standings.sort_values('Points', ascending=False).reset_index(drop=True)
        
        return final_standings, constructor_standings, race_results
    
    def train_model(self):
        # Get prepared features from data loader
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.prepare_features()
        
        # Initialize and train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)
        
        # Calculate probabilities for ROC AUC
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]
        test_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Train Accuracy': accuracy_score(y_train, train_pred),
            'Validation Accuracy': accuracy_score(y_val, val_pred),
            'Test Accuracy': accuracy_score(y_test, test_pred),
            'Train ROC AUC': roc_auc_score(y_train, train_proba),
            'Validation ROC AUC': roc_auc_score(y_val, val_proba),
            'Test ROC AUC': roc_auc_score(y_test, test_proba)
        }
        
        # Calculate precision, recall, and F1 score for test set
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='binary')
        metrics.update({
            'Test Precision': precision,
            'Test Recall': recall,
            'Test F1 Score': f1
        })
        
        return metrics
    
    def save_model(self, filename='f1_model.joblib'):
        if self.model is not None:
            model_data = {
                'model': self.model,
                'feature_importance': self.feature_importance,
                'label_encoders': self.data_loader.label_encoders
            }
            joblib.dump(model_data, filename)
            return f"Model saved to {filename}"
    
    def load_model(self, filename='f1_model.joblib'):
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.data_loader.label_encoders = model_data['label_encoders']
        return f"Model loaded from {filename}"

def main():
    st.set_page_config(
        page_title="F1 2025 Race Winner Predictor",
        page_icon="🏎️",
        layout="wide"
    )
    
    st.title('🏎️ F1 2025 Race Winner Predictor')
    st.write('Predicting Formula 1 race winners using machine learning')
    
    predictor = F1Predictor()
    
    # Sidebar
    st.sidebar.header('Model Controls')
    
    # Main content area tabs
    tab1, tab2, tab3 = st.tabs(["Model Training", "2025 Predictions", "Championship Prediction"])
    
    with tab1:
        if st.button('Train New Model'):
            with st.spinner('Training model...'):
                try:
                    metrics = predictor.train_model()
                    
                    # Display metrics in main area
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader('Accuracy Metrics')
                        for metric in ['Train Accuracy', 'Validation Accuracy', 'Test Accuracy']:
                            st.metric(metric, f"{metrics[metric]:.2%}")
                    
                    with col2:
                        st.subheader('ROC AUC Scores')
                        for metric in ['Train ROC AUC', 'Validation ROC AUC', 'Test ROC AUC']:
                            st.metric(metric, f"{metrics[metric]:.2%}")
                    
                    with col3:
                        st.subheader('Test Set Metrics')
                        for metric in ['Test Precision', 'Test Recall', 'Test F1 Score']:
                            st.metric(metric, f"{metrics[metric]:.2%}")
                    
                    # Feature importance plot
                    st.subheader('Feature Importance')
                    fig = px.bar(
                        predictor.feature_importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Model Feature Importance'
                    )
                    fig.update_layout(
                        xaxis_title='Importance Score',
                        yaxis_title='Feature',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save the model
                    save_message = predictor.save_model()
                    st.success('Model trained and saved successfully!')
                    st.info(save_message)
                    
                except Exception as e:
                    st.error(f'Error during model training: {str(e)}')
    
    with tab2:
        if predictor.model is None:
            st.warning("Please train a model first or load an existing model.")
            if st.button("Load Existing Model"):
                try:
                    load_message = predictor.load_model()
                    st.success(load_message)
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            st.subheader("2025 Race Predictions")
            
            # Add circuit selection from calendar
            circuit = st.selectbox(
                "Select Circuit",
                F1_CALENDAR_2025
            )
            
            # Option to modify grid positions
            modify_grid = st.checkbox("Modify Grid Positions")
            
            qualifying_results = None
            if modify_grid:
                st.write("Enter grid positions (1-20):")
                col1, col2 = st.columns(2)
                qualifying_results = {}
                
                for idx, row in predictor.grid_2025.iterrows():
                    if idx % 2 == 0:
                        with col1:
                            pos = st.number_input(
                                f"{row['driver_name']} ({row['team_name']})",
                                min_value=1,
                                max_value=20,
                                value=idx + 1
                            )
                    else:
                        with col2:
                            pos = st.number_input(
                                f"{row['driver_name']} ({row['team_name']})",
                                min_value=1,
                                max_value=20,
                                value=idx + 1
                            )
                    qualifying_results[row['driverId']] = pos
            
            if st.button("Predict Race Results"):
                results = predictor.predict_2025_race(circuit, qualifying_results)
                
                if results is not None:
                    st.write(f"Predicted Race Results for {circuit}")
                    
                    # Create a more visually appealing results table
                    fig = go.Figure(data=[
                        go.Table(
                            header=dict(
                                values=['Position', 'Driver', 'Team', 'Grid', 'Win Probability', 'Championship Points'],
                                fill_color='darkblue',
                                align='left',
                                font=dict(color='white', size=12)
                            ),
                            cells=dict(
                                values=[
                                    list(range(1, len(results) + 1)),  # Convert range to list
                                    results['Driver'],
                                    results['Team'],
                                    results['Grid'],
                                    [f"{x:.1%}" for x in results['Win Probability']],
                                    results['Championship Points']
                                ],
                                align='left',
                                font=dict(size=11),
                                height=30
                            )
                        )
                    ])
                    
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a bar chart of win probabilities
                    prob_fig = px.bar(
                        results.head(10),
                        x='Driver',
                        y='Win Probability',
                        title='Top 10 Drivers - Win Probability',
                        text=[f"{x:.1%}" for x in results.head(10)['Win Probability']]
                    )
                    prob_fig.update_traces(textposition='outside')
                    prob_fig.update_layout(height=400)
                    st.plotly_chart(prob_fig, use_container_width=True)

    with tab3:
        if predictor.model is None:
            st.warning("Please train a model first or load an existing model.")
            if st.button("Load Existing Model", key="load_model_championship"):
                try:
                    load_message = predictor.load_model()
                    st.success(load_message)
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            st.subheader("2025 Championship Prediction")
            
            if st.button("Simulate Remaining Races"):
                with st.spinner("Simulating championship..."):
                    driver_standings, constructor_standings, race_results = predictor.simulate_championship()
                    
                    # Display Driver's Championship
                    st.subheader("Predicted Driver's Championship Standings")
                    fig_drivers = go.Figure(data=[
                        go.Table(
                            header=dict(
                                values=['Position', 'Driver', 'Team', 'Points'],
                                fill_color='darkblue',
                                align='left',
                                font=dict(color='white', size=12)
                            ),
                            cells=dict(
                                values=[
                                    list(range(1, len(driver_standings) + 1)),
                                    driver_standings['Driver'],
                                    driver_standings['team_name'],
                                    driver_standings['Points']
                                ],
                                align='left',
                                font=dict(size=11),
                                height=30
                            )
                        )
                    ])
                    fig_drivers.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
                    st.plotly_chart(fig_drivers, use_container_width=True)
                    
                    # Display Constructor's Championship
                    st.subheader("Predicted Constructor's Championship Standings")
                    fig_constructors = go.Figure(data=[
                        go.Table(
                            header=dict(
                                values=['Position', 'Team', 'Points'],
                                fill_color='darkblue',
                                align='left',
                                font=dict(color='white', size=12)
                            ),
                            cells=dict(
                                values=[
                                    list(range(1, len(constructor_standings) + 1)),
                                    constructor_standings['team_name'],
                                    constructor_standings['Points']
                                ],
                                align='left',
                                font=dict(size=11),
                                height=30
                            )
                        )
                    ])
                    fig_constructors.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
                    st.plotly_chart(fig_constructors, use_container_width=True)
                    
                    # Display Race Winners
                    st.subheader("Predicted Race Winners")
                    race_results_df = pd.DataFrame(race_results)
                    fig_races = go.Figure(data=[
                        go.Table(
                            header=dict(
                                values=['Race', 'Winner', 'Second', 'Third'],
                                fill_color='darkblue',
                                align='left',
                                font=dict(color='white', size=12)
                            ),
                            cells=dict(
                                values=[
                                    race_results_df['Race'],
                                    race_results_df['Winner'],
                                    race_results_df['Second'],
                                    race_results_df['Third']
                                ],
                                align='left',
                                font=dict(size=11),
                                height=30
                            )
                        )
                    ])
                    fig_races.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=800)
                    st.plotly_chart(fig_races, use_container_width=True)
                    
                    # Add points visualization
                    st.subheader("Championship Points Distribution")
                    points_fig = px.bar(
                        driver_standings.head(10),
                        x='Driver',
                        y='Points',
                        title='Top 10 Drivers - Championship Points',
                        color='team_name',
                        text='Points'
                    )
                    points_fig.update_traces(textposition='outside')
                    points_fig.update_layout(height=400)
                    st.plotly_chart(points_fig, use_container_width=True)

if __name__ == "__main__":
    main()

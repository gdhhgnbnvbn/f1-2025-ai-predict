import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class F1DataLoader:
    def __init__(self, data_path='f1data'):
        self.data_path = data_path
        self.data = {}
        self.label_encoders = {}
    
    def load_all_data(self):
        """Load all necessary CSV files."""
        files = [
            'races', 'results', 'drivers', 'constructors',
            'qualifying', 'circuits', 'driver_standings',
            'constructor_standings'
        ]
        
        for file in files:
            self.data[file] = pd.read_csv(f'{self.data_path}/{file}.csv')
            
        # Ensure numeric types for critical columns
        if 'points' in self.data['results'].columns:
            self.data['results']['points'] = pd.to_numeric(self.data['results']['points'], errors='coerce')
        if 'position' in self.data['results'].columns:
            self.data['results']['position'] = pd.to_numeric(self.data['results']['position'], errors='coerce')
        
        return self.data
    
    def prepare_race_data(self):
        """Prepare and merge race data with proper handling of duplicate columns."""
        # Start with results and add race information
        df = pd.merge(
            self.data['results'],
            self.data['races'][['raceId', 'year', 'round', 'circuitId']],
            on='raceId'
        )
        
        # Add driver information
        df = pd.merge(
            df,
            self.data['drivers'][['driverId', 'nationality']],
            on='driverId',
            suffixes=(None, '_driver')
        )
        
        # Add constructor information
        df = pd.merge(
            df,
            self.data['constructors'][['constructorId', 'nationality']],
            on='constructorId',
            suffixes=(None, '_constructor')
        )
        
        # Add circuit information (explicitly handle duplicate columns)
        circuit_columns = ['circuitId', 'name', 'location', 'country']
        df = pd.merge(
            df,
            self.data['circuits'][circuit_columns],
            on='circuitId',
            suffixes=('', '_circuit')
        )
        
        return df
    
    def add_features(self, df):
        """Add engineered features to the dataset."""
        # Ensure numeric types
        numeric_columns = ['points', 'position', 'grid']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Recent performance (last 3 races)
        df['points_moving_avg'] = df.sort_values('raceId').groupby('driverId')['points'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Qualifying performance
        qual_data = pd.merge(
            self.data['qualifying'],
            self.data['races'][['raceId', 'year']],
            on='raceId'
        )
        qual_data['position'] = pd.to_numeric(qual_data['position'], errors='coerce')
        qual_data['qual_position_avg'] = qual_data.sort_values('raceId').groupby('driverId')['position'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df = pd.merge(
            df,
            qual_data[['raceId', 'driverId', 'qual_position_avg']],
            on=['raceId', 'driverId'],
            how='left'
        )
        
        # Circuit performance
        df['circuit_wins'] = df.sort_values('raceId').groupby(['driverId', 'circuitId'])['position'].transform(
            lambda x: (x == 1).cumsum()
        )
        
        # Season performance
        df = pd.merge(
            df,
            self.data['driver_standings'][['raceId', 'driverId', 'points', 'position']],
            on=['raceId', 'driverId'],
            suffixes=('', '_championship')
        )
        
        # Constructor performance
        constructor_stats = df.groupby('constructorId').agg({
            'points': ['mean', 'std'],
            'position': 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        constructor_stats.columns = ['constructorId', 'constructor_points_mean', 
                                   'constructor_points_std', 'constructor_position_mean']
        
        # Fill NaN values with appropriate defaults
        constructor_stats = constructor_stats.fillna({
            'constructor_points_mean': 0,
            'constructor_points_std': 0,
            'constructor_position_mean': df['position'].max()
        })
        
        df = pd.merge(df, constructor_stats, on='constructorId')
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables."""
        categorical_columns = ['nationality', 'nationality_constructor', 'country']
        
        for col in categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def prepare_features(self):
        """Prepare all features for model training."""
        # Load all data
        self.load_all_data()
        
        # Prepare base dataset
        df = self.prepare_race_data()
        
        # Add engineered features
        df = self.add_features(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df)
        
        # Create target variable
        df['winner'] = (df['position'] == 1).astype(int)
        
        # Select features for the model
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
        
        # Ensure all feature columns exist and handle missing values
        for col in feature_columns:
            if col not in df.columns:
                print(f"Warning: Column {col} not found in dataset")
            else:
                df[col] = df[col].fillna(df[col].mean() if df[col].dtype.kind in 'iuf' else df[col].mode()[0])
        
        # Split data by year
        train_data = df[df['year'] <= 2022]
        val_data = df[df['year'] == 2023]
        test_data = df[df['year'] == 2024]
        
        # Return only the necessary columns
        return (
            train_data[feature_columns], train_data['winner'],
            val_data[feature_columns], val_data['winner'],
            test_data[feature_columns], test_data['winner']
        )

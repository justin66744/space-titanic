# space_titanic.py
#
# ICS 32 
# Project #4: Spaceship Titanic
#
# Machine Learning Pipeline for Kaggle's Spaceship Titanic competition.
# 
# NAME: Justin Tran
# EMAIL: justitt6@uci.edu
# STUDENT ID: 77585452

from pathlib import Path
import time
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm


SEED = 1234

class TitanicModel:
    '''
    TitanicModel
    '''
    def __init__(self, competition_name: str, download_path: str = "data", file_name: str = "submission.csv"):
        self.competition_name = competition_name
        self.download_path = Path(download_path)
        self.file_name = file_name
        self.api = KaggleApi()
        self.api.authenticate()
        self.train_data = None
        self.test_data = None
        self.best_model = None
        self.best_score = 0

    # Data Acquisition
    def download_dataset(self) -> None:
        """Downloads and extracts Kaggle dataset."""
        self.download_path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset: {self.competition_name}...")
        self.api.competition_download_files(self.competition_name, self.download_path)

        zip_path = self.download_path / f"{self.competition_name}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.download_path)
            zip_path.unlink()
        else:
            print(f"Error: {zip_path} not found.")

    def load_data(self) -> None:
        """
        Loads training and test datasets.
        First, checks to see if train.csv exists in the path used by download_dataset.
        If not, call download_dataset() to download data from Kaggle.
        Read train.csv and store returned data frame in self.train_data.
        Read test.csv and store returned data frame in self.test_data.
        """
        train_file = self.download_path / "train.csv"
        if not train_file.exists():
            self.download_dataset()
        
        self.train_data = pd.read_csv(train_file)

        test_file = self.download_path / "test.csv"

        if not test_file.exists():
            return

        self.test_data = pd.read_csv(test_file)

    # Data Preprocessing

    @staticmethod
    def preprocess(data: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values in the dataset with the median of each column."""
        data.fillna(data.median(numeric_only=True), inplace=True)
        return data


    # Model Training
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Trains and selects the best KNN model using hyperparameter tuning."""
        training_scores = {}

        # Iterate over different values of n_neighbors (k)
        for n_neighbors in tqdm(range(1, 51), desc="Tuning k-NN Hyperparameters"):  
            # Initialize KNN
            model = knn(n_neighbors=n_neighbors)

            # Train KNN model
            # figure out what .fit takes as arguments
            model.fit(X_train, y_train)

            # Calculate accuracy on validation data
            # figure out what accuracy_score takes as arguments
            #       only need to pass first two arguments
            y_predict = model.predict(X_val)

            score = accuracy_score(y_val, y_predict)

            training_scores[n_neighbors] = score  # Store the accuracy value directly

            # Track the best model based on accuracy
            if score > self.best_score:
                # assign model to best_model instance variable
                # assign score to best_score instance variable
                self.best_model = model
                self.best_score = score

        # add  best_model and best_score instance variables to print
        print(f"Best model {self.best_model.n_neighbors} achieved validation accuracy: {self.best_score:.2f} \n")
        self.visualize_training_process(training_scores)


    # Visualization
    @staticmethod
    def visualize_data(data: pd.DataFrame) -> None:
        """Visualizes missing data and feature correlations."""
        # Visualize missing data
        print(f"\n   Sample of data: \n\n {data.head()}")
        print(f"\n   Shape of data: {data.shape}")
        data.info()
        print(data.describe())

        missing = data.isnull().sum()
        columns = list(missing.index)
        counts = list(missing.values)

        # create bar graph of size 12, 8
        # Save the figure in missing.png 
        # with columns as x values and counts as y values
        # use the following code to avoid overlapping x-axis labels 
        # plt.xticks(rotation=45)
        # start a new figure with "plt.figure(figsize=(12, 8))"
        plt.figure(figsize=(12, 8))
        plt.figure(figsize=(12, 8))

        plt.bar(columns, counts, color='skyblue')
        plt.title("Missing Values")
        plt.ylabel("# of Missing Values")
        plt.xlabel("Cols")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')

        plt.tight_layout()
        plt.savefig("missing.png")
        plt.show()

    @staticmethod
    def visualize_correlation(data: pd.DataFrame) -> None:
        '''
        Creates graphs
        '''
        # Calculate correlation with 'Transported'
        correlation = data.corrwith(data['Transported'])
        correlation = correlation.drop('Transported').sort_values(ascending=False)
        columns = list(correlation.index)
        counts = list(correlation.values)

        # Plot the correlation in a 10, 10 figure
        # Save in correlation.png
        # Use plt.xticks(rotation=) to avoid overlap 
        # X-axis label is Features, and y-axis label is Correlation

        correlation = data.corrwith(data['Transported'])
        correlation = correlation.drop('Transported').sort_values(ascending=False)
        columns = list(correlation.index)
        counts = list(correlation.values)

        plt.figure(figsize=(10, 10))
        plt.bar(columns, counts, color='purple')
        plt.title("Feature Correlation with Transported")
        plt.ylabel("Correlation")
        plt.xlabel("Features")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')

        plt.tight_layout()
        plt.savefig("correlation.png")
        plt.show()

    @staticmethod
    def visualize_training_process(training_scores: dict[int, float]) -> None:
        """Visualizes training and validation accuracy during hyperparameter tuning for k-NN."""
        # Plot accuracy for each value of n_neighbors
        # Save plot in training_knn.png"
        # training_scores is a dictionary where 
        # key = n_neighbors and value = score 
        n_neighbors = list(training_scores.keys())
        accuracy_scores = list(training_scores.values())

        plt.figure(figsize=(10, 6))
        plt.plot(n_neighbors, accuracy_scores, marker='o', linestyle='-', color='blue', label='Validation Accuracy')
        plt.title("k-NN Tuning: Validation Accuracy vs. # of Neighbors")
        plt.ylabel("Validation Accuracy")
        plt.xlabel("# of k-Neighbors")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(n_neighbors)
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_knn.png")
        plt.show()

    @staticmethod
    def encode(data: pd.DataFrame) -> None:
        # Have to clean the data and encode strings into numbers
        for col in data.select_dtypes(include="object"):
            if col != "PassengerId": # problem with passenger id column
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # Submission
    def create_submission_file(self, predictions: pd.Series, passenger_ids: pd.Series) -> None:
        """Saves predictions in Kaggle submission format."""
        submission = pd.DataFrame({
            "PassengerId": passenger_ids,
            "Transported": predictions
        })
        submission.to_csv(self.file_name, index=False)
        print(f"Submission saved to {self.file_name}")

    def submit(self, competition_name: str, file_name: str) -> None:
        '''
        Submits project
        '''
        self.api.competition_submit(file_name, message="Final Predictions", competition=competition_name)
        print("Submission successful!")

        time.sleep(10)
        submissions = self.api.competition_submissions(competition_name)

        for idx, submission in enumerate(submissions):
            print(f"Submission ID: {submission.ref}")
            print(f"Public Score: {submission.publicScore}")
            print("-" * 40)
            if idx == 3:
                break

        return submissions
    
    # Main Pipeline
    def run(self) -> None:
        """Executes the entire ML pipeline."""
        # Step 1: Data Acquisition
        self.load_data()

        # Step 2: Visualize Missing Data
        self.visualize_data(self.train_data)

        # Step 3: Encode Data
        self.encode(self.train_data)
        self.encode(self.test_data)

        # Step 4: Visualize Correlation
        self.visualize_correlation(self.train_data)

        # Step 5: Data Preprocessing

        self.train_data = self.preprocess(self.train_data)
        self.test_data = self.preprocess(self.test_data)

        # Drop rows with missing values in 'Transported'
        self.train_data = self.train_data.dropna(subset=["Transported"])

        # Prepare Data for Training
        X = self.train_data.drop(columns=["Transported"])
        y = self.train_data["Transported"]

        # Split dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=SEED)

        # Step 6: Baseline KNN Model Training - uses all features
        # This baseline KNN model should achieve validation accuracy of ~58%
        selected_features = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
                            'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                            'Name']

        # Train model on selected features
        X_train_top_features = X_train[selected_features]
        X_val_top_features = X_val[selected_features]

        self.train_model(X_train_top_features, y_train, X_val_top_features,  y_val)          

        # Step 7: Feature Selection (manual update by the student)
        print("Student should update the selected_features list with desired features.")
        print("Available features to select from:")
        print(X_train.columns)

        #       dentify and include most relevant features to train on
        #       full list of features is provided in Step 6.

        selected_features = ['CryoSleep', 'Age', 'RoomService', 'Spa', 'VRDeck',
                             'HomePlanet', 'Destination', 'VIP', 'FoodCourt', 'ShoppingMall']

        # Train model on selected features
        X_train_top_features = X_train[selected_features]
        X_val_top_features = X_val[selected_features]

        self.train_model(X_train_top_features, y_train, X_val_top_features,  y_val)

        # Step 8: Prediction
        passenger_ids = self.test_data["PassengerId"]
        self.test_data = self.test_data[X_train_top_features.columns]
        predictions = self.best_model.predict(self.test_data)

        # Step 9: Submission

        self.create_submission_file(predictions, passenger_ids)
        #self.submit("spaceship-titanic", "submission.csv")

if __name__ == "__main__":
    titanic_model = TitanicModel(competition_name="spaceship-titanic")
    titanic_model.run()

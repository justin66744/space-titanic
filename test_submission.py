import unittest
import pandas as pd
from pathlib import Path
from space_titanic import TitanicModel

class TestTitanicModel(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.download_path = Path("temp_test_data")

        self.model = TitanicModel("spaceship-titanic", download_path=str(self.download_path))

        self.train_data = pd.DataFrame({
            "Age": [25, 30, None, 22, 28, 35, 40, 45, 50, 55] * 6,  # 60 samples
            "RoomService": [100, None, 200, 150, 50, 300, 400, 500, 600, 700] * 6,
            "Spa": [None, 150, 100, 200, None, 250, 300, 350, 400, 450] * 6,
            "HomePlanet": ["Earth", "Mars", "Earth", "Mars", "Earth", "Mars", "Earth", "Mars", "Earth", "Mars"] * 6,
            "Transported": [True, False, True, False, True, False, True, False, True, False] * 6
        })

    def test_load_data(self):
        """Test if load_data correctly loads train and test data when files exist."""
        train_file = self.download_path / "train.csv"
        test_file = self.download_path / "test.csv"

        self.model.load_data()

        self.assertIsNotNone(self.model.train_data, "Train data was not loaded")
        self.assertIsNotNone(self.model.test_data, "Test data was not loaded")

    def test_preprocess(self):
        """Test the preprocess method to ensure missing values are filled."""
        processed_data = self.model.preprocess(self.train_data)
        self.assertFalse(processed_data.isnull().any().any(), "Missing values were not filled")

    def test_train_model(self):
        """Test the train_model method to ensure it trains a model and selects the best one."""
        self.train_data = self.model.preprocess(self.train_data)
        self.model.encode(self.train_data)

        X = self.train_data.drop(columns=["Transported"])
        y = self.train_data["Transported"]

        X_train = X.iloc[:50]
        y_train = y.iloc[:50]
        X_val = X.iloc[50:]
        y_val = y.iloc[50:]

        self.model.train_model(X_train, y_train, X_val, y_val)

        self.assertIsNotNone(self.model.best_model, "Not the best model")
        self.assertGreater(self.model.best_score, 0, "Not the best score")

    def test_visualize_data(self):
        """Test the visualize_data method to ensure it doesn't raise exceptions."""
        try:
            self.model.visualize_data(self.train_data)
            success = True
        except Exception as e:
            success = False
            self.fail(f"visualize_data exception: {e}")
        
        self.assertTrue(success)

    def test_visualize_correlation(self):
        """Test the visualize_correlation method to ensure it doesn't raise exceptions."""
        numeric_data = pd.DataFrame({
        "Age": [25, 30, 22, 28, 35, 40, 45, 50, 55, 60],
        "RoomService": [100, 200, 150, 50, 300, 400, 500, 600, 700, 800],
        "Spa": [150, 100, 200, 250, 300, 350, 400, 450, 500, 550],
        "Transported": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                                    })
        try:
            self.model.visualize_correlation(numeric_data)
            success = True
        except Exception as e:
            success = False
            self.fail(f"visualize_correlation exception: {e}")
        
        self.assertTrue(success)

    def test_visualize_training_process(self):
        """Test the visualize_training_process method with sample training scores."""
        training_scores = {1: 0.6, 2: 0.65, 3: 0.7, 4: 0.68, 5: 0.72}
        
        try:
            self.model.visualize_training_process(training_scores)
            success = True
        except Exception as e:
            success = False
            self.fail(f"visualize_training_process exception: {e}")
        
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()

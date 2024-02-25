import pandas as pd
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class CSVReader:
    """
    A class to read and store CSV files from a directory.

    Attributes:
    - directory (str): The path to the directory containing CSV files.
    - dataframes (dict): A dictionary to store DataFrames with file names as keys.

    Methods:
    - read_and_save_csv(): Reads CSV files from the directory and saves them as DataFrames.
    - print_all_dataframes(): Prints all stored DataFrames along with their file names.
    """

    def __init__(self, directory):
        """
        Initializes the CSVReader class.

        Args:
        - directory (str): The path to the directory containing CSV files.
        """
        self.directory = directory
        self.dataframes = {}

    def read_and_save_csv(self):
        """
        Reads CSV files from the directory and saves them as DataFrames.
        """
        # List all files in the directory
        files = os.listdir(self.directory)

        # Iterate through each file
        for file in files:
            if file.endswith('.csv'):
                # Read the CSV file
                file_path = os.path.join(self.directory, file)
                df = pd.read_csv(file_path)

                # Save DataFrame with file name as key
                self.dataframes[file] = df
        return self.dataframes

    def print_all_dataframes(self):
        """
        Prints all stored DataFrames along with their file names.
        """
        # Display all DataFrames
        for file, df in self.dataframes.items():
            print("File:", file)
            print(df)


# Usage
directory = "C:\\Users\\baspi\\OneDrive\\Masaüstü\\MaterialData\\DataFile"
csv_reader = CSVReader(directory)
df = csv_reader.read_and_save_csv()

import pandas as pd

from utils import split_into_train_test


class Dataset:
    def __init__(self, csv):
        self.x, self.y = self.load_dataset(csv)
        self.x_train, self.x_test, self.y_train, self.y_test = split_into_train_test(self.x, self.y, 0.3)

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass


class BreastTissueDataset(Dataset):
    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 10 attributes: 9 features + 1 class attribute under "Class" column
        df = pd.read_csv("../data/extracted/breast_tissue.csv", delimiter=";")
        x = df.drop(['Case #', 'Class'], axis='columns')
        y = df['Class']
        return x, y

class BreastTissueDataset(Dataset):
    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 10 attributes: 9 features + 1 class attribute under "Class" column
        df = pd.read_csv("../data/extracted/breast_tissue.csv", delimiter=";")
        x = df.drop(['Case #', 'Class'], axis='columns')
        y = df['Class']
        return x, y
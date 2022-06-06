import pandas as pd


class Dataset:
    name = ""

    def __init__(self, csv):
        self.x, self.y = self.load_dataset(csv)

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise Exception("This is an abstract method")


class BreastTissueDataset(Dataset):
    name = "Breast Tissue Dataset"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 10 attributes: 9 features + 1 class attribute under "Class" column
        df = pd.read_csv(csv)
        x = df.drop(['Case #', 'Class'], axis='columns')
        y = df['Class']
        return x, y


class CarEvaluationDataset(Dataset):
    name = "Car Evaluation Dataset"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 1728 instances
        # acceptability - last column
        df = pd.read_csv(csv)
        x = df.drop(['acceptability'], axis='columns')
        x = pd.get_dummies(x)
        y = df['acceptability']
        return x, y


class TitanicDataset(Dataset):
    name = "Titanic Dataset"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(csv)
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis='columns')
        df = pd.get_dummies(df)
        df.dropna(inplace=True)
        x = df.drop(['Survived'], axis='columns')
        y = df['Survived']
        return x, y


class WineDataset(Dataset):
    name = "Wine Dataset"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(csv)
        x = df.drop(['Class'], axis='columns')
        y = df['Class']
        return x, y


class RedWineQualityDataset(Dataset):
    name = "Red Wine Quality Dataset"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(csv)
        x = df.drop(['quality'], axis='columns')
        y = df['quality']
        return x, y

import pandas as pd


class Dataset:
    name = ""

    def __init__(self, x, y):
        self.x = pd.DataFrame(x)
        self.y = pd.Series(y)


class CSVDataset(Dataset):
    name = ""

    def __init__(self, csv):
        super().__init__(*self.load_dataset(csv))

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise Exception("This is an abstract method")


class BreastTissueDataset(CSVDataset):
    name = "breast_tissue"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 10 attributes: 9 features + 1 class attribute under "Class" column
        df = pd.read_csv(csv)
        x = df.drop(['Case #', 'Class'], axis='columns')
        y = df['Class']
        return x, y


class CarEvaluationDataset(CSVDataset):
    name = "car_evaluation"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 1728 instances
        # acceptability - last column
        df = pd.read_csv(csv)
        x = df.drop(['acceptability'], axis='columns')
        x = pd.get_dummies(x)
        y = df['acceptability']
        return x, y


class TitanicDataset(CSVDataset):
    name = "titanic"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(csv)
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis='columns')
        df = pd.get_dummies(df)
        df.dropna(inplace=True)
        x = df.drop(['Survived'], axis='columns')
        y = df['Survived']
        return x, y


class WineDataset(CSVDataset):
    name = "wine"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(csv)
        x = df.drop(['Class'], axis='columns')
        y = df['Class']
        return x, y


class RedWineQualityDataset(CSVDataset):
    name = "red_wine_quality"

    @staticmethod
    def load_dataset(csv) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(csv)
        x = df.drop(['quality'], axis='columns')
        y = df['quality']
        return x, y

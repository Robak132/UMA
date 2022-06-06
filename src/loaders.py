import pandas

# TODO: Remove legacy code


def load_dataset(path):
    return pandas.read_csv(path)


def load_breast_tissue():
    # 106 instances
    # 10 attributes: 9 features + 1 class attribute under "Class" column
    df = pandas.read_csv("../data/extracted/breast_tissue.csv")
    x = df.drop(['Case #', 'Class'], axis='columns')
    y = df['Class']
    return x, y


def load_car_evaluation():
    df = pandas.read_csv("../data/extracted/car_evaluation.csv")
    return df


def load_titanic():
    df = pandas.read_csv('../data/extracted/titanic.csv')
    df.drop('PassengerId', axis='columns', inplace=True)
    return df


def load_wine():
    df = pandas.read_csv('../data/extracted/wine.csv')
    return df


def load_wine_quality_red():
    df = pandas.read_csv('../data/extracted/winequality_red.csv')
    return df

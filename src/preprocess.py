import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(input_path, output_train, output_test):
    df = pd.read_csv(input_path, header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv(output_train, index=False)
    test.to_csv(output_test, index=False)

if __name__ == "__main__":
    preprocess_data("data/iris.csv", "data/train.csv", "data/test.csv")
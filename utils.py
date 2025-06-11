from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_dataset(dataset_path):
    # Tải dataset từ local CSV
    dataset = load_dataset("csv", data_files=dataset_path)

    # Chuẩn hóa label thành số (0,1,2,3)
    le = LabelEncoder()

    # Fit LabelEncoder once on the entire dataset instead of per example
    labels = dataset["train"]["label"]
    le.fit(labels)

    # Transform each example using the fitted encoder
    dataset = dataset.map(lambda example: {
        "label": int(le.transform([example["label"]])[0])
    })

    return dataset, le

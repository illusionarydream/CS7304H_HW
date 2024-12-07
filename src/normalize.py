from sklearn.preprocessing import MaxAbsScaler
import pickle

train_data = pickle.load(open("datasets/train_feature.pkl", "rb"))
test_data = pickle.load(open("datasets/test_feature.pkl", "rb"))

scaler = MaxAbsScaler()
normalized_train_data = scaler.fit_transform(train_data)
normalized_test_data = scaler.transform(test_data)

pickle.dump(normalized_train_data, open(
    "datasets/normalized_train_feature.pkl", "wb"))
pickle.dump(normalized_test_data, open(
    "datasets/normalized_test_feature.pkl", "wb"))

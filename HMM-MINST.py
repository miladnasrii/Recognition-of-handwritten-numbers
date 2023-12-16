import numpy as np
from hmmlearn import hmm
from sklearn import datasets

numbers = datasets.load_digits()
data = numbers.data
real_numbers = numbers.target

model = hmm.GaussianHMM(n_components=10, covariance_type="full")

model.fit(data)

predicted_num = model.predict(data)

print("Predicted numbers :", predicted_num)

assessment = np.mean(predicted_num == real_numbers)
print("Assessment :", assessment)
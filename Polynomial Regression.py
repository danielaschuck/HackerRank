import numpy as np 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression

f, n = map(int, input().strip().split())
    
x_observations = [];
y_amount = []
for _ in range(n):
    l = list(map(float, input().strip().split()))
    x_observations.append(l[0:-1])
    y_amount.append(l[-1])

poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(np.array(x_observations))

model = LinearRegression()
model.fit(X_poly, np.array(y_amount))

#result
t = int(input().strip())
for _ in range(t):
    l = poly_features.fit_transform(np.array(list(map(float, input().strip().split()))).reshape(1, -1))
    print( round(( model.predict(l) )[0],2))
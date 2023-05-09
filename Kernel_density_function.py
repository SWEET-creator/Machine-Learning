from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,
random_state=1, stratify=y)

sc = StandardScaler()
sc = sc.fit(X_train)
X_train_std = sc.transform(X_tarin)
X_test_std = sc.transtorm(X_test)

kernel0 = gaussian_kde(X_train_std[y_train==0].T)
kernel1 = gaussian_kde(X_train_std[y_train==1].T)
kernel2 = gaussian_kde(X_train_std[y_train==2].T)

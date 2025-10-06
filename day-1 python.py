import numpy as np, pandas as pd

# --- NumPy Practice ---
a=np.array([1,2,3,4,5]); b=np.array([[1,2,3],[4,5,6]])
print("NumPy mean:",a.mean(),"Dot:",np.dot(a,a),"Norm:",np.linalg.norm(a),"Shape:",b.shape)

# --- Pandas Practice ---
df=pd.DataFrame({"Name":["Alice","Bob"],"Age":[25,30],"Score":[85,90]})
print("Pandas head:\n",df.head(),"\nDescribe:\n",df.describe())

# --- Probability Practice ---
x=np.random.rand(10); y=np.random.randn(10)
print("Prob mean:",np.mean(y),"Var:",np.var(y),"Std:",np.std(y))

# --- Linear Algebra Practice ---
A=np.array([[1,2],[3,4]]); v=np.array([5,6])
print("A·v:",np.dot(A,v),"Transpose:\n",A.T,"Inverse:\n",np.linalg.inv(A))

# --- Linear Regression (Normal Equation) ---
np.random.seed(42)
X=np.random.rand(100,1); y=3*X.squeeze()+2+np.random.randn(100)*0.2
X_b=np.c_[np.ones((100,1)),X]; theta=np.linalg.inv(X_b.T@X_b)@X_b.T@y
print("Regression theta:",theta)

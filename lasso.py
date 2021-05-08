#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200
import cvxpy as cp


#%%
n, p = 3000, 1000
A = 3.5
k = 35
beta = np.zeros(p)
beta[:k] = A

X = np.random.randn(n,p)/np.sqrt(n)
y = X@beta + np.random.randn(n)

r = np.abs(X.T @ y)
plt.hist(r)
plt.show()

#%%

plt.hist(r)
plt.savefig('Xty')

#%%

lambd = 1.6
betahat = cp.Variable(shape=p)
obj = 0.5*cp.norm2(y - X @ betahat)**2 + lambd*cp.norm(betahat, 1)
prob = cp.Problem(cp.Minimize(obj))
prob.solve()
print("status: {}".format(prob.status))
# Number of nonzero elements in the solution (its cardinality or diversity).
nnz_l1 = (np.absolute(betahat.value) > 1e-4).sum()
print('Found a feasible beta hat in R^{} that has {} nonzeros.'.format(p, nnz_l1))
print('Relative error: {}'.format(np.linalg.norm(betahat.value-beta)/np.linalg.norm(beta)))


#%%

plt.plot(betahat.value[:70],'*')
plt.savefig('lasso')

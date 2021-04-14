#%%

import numpy as np
from scipy.stats import chi2
from scipy.stats import norm

#%%


class Test:
    def __init__(self, n=1_000_000, alpha=0.05):
        self.n = n
        self.alpha = alpha
        self.thresh = {'Bonferroni': norm.ppf(1 - alpha / 2 / n),
                       'Chi2': chi2.ppf(1 - alpha, df=n),
                       'Fisher': chi2.ppf(1 - alpha, df=2 * n)}

    def theta(self, mu):
        return np.linalg.norm(mu) ** 2 / np.sqrt(2 * self.n)

    def sampleY(self, mu, B):
        self.Y = mu[np.newaxis, :] + np.random.randn(B, n)

    def testValue(self):
        y = self.Y
        pvals = 2 * norm.cdf(-np.abs(y))
        return {'Bonferroni': np.abs(y).max(axis=1) > self.thresh['Bonferroni'],
                'Chi2': np.sum(y**2, axis=1) > self.thresh['Chi2'],
                'Fisher': np.sum(-2 * np.log(pvals), axis=1) > self.thresh['Fisher']}

    def print_info(self, mu):
        print('Value of Bonferroni Threshold: {:.2f}'.format(
            self.thresh['Bonferroni']))
        print(
            'Value of theta = ||mu||^2/sqrt(2n): {:.2f}'.format(self.theta(mu)))
        print(
            'Expected power of chi-sq test: {:.2f}'.format(norm.cdf(self.theta(mu) - 1.645)))

    def print_power(self, T):
        for method in T.keys():
            power = T[method].mean()
            print(method + ' power: {:.1f}'.format(100 * power))

#%% Sparse strong effects


n = 1_000_000
alpha = 0.05
results = Test(n, alpha)

mu = np.zeros(n)
mu[:4] = results.thresh['Bonferroni']
results.print_info(mu)

#%%

results.sampleY(mu, B=50)
T = results.testValue()
results.print_power(T)

#%% Distributed (weaker) effects

k = 2400
mu = np.zeros(n)
mu[:k] = 1.1
results.print_info(mu)

#%%

results.sampleY(mu, B=50)
T = results.testValue()
results.print_power(T)

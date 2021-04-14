#%%

import numpy as np
from scipy.stats import ncx2
from scipy.stats import norm
from scipy.stats import chi2


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

#%%

df = 10_000
theta = 2
nc = np.sqrt(2*df)*theta

x = np.linspace(ncx2.ppf(0.001, df, nc),
                ncx2.ppf(0.999, df, nc), 1000)

plt.plot(x, ncx2.pdf(x, df, nc),label='theta = {:.2f}'.format(theta))
plt.legend()
plt.show()

#%%

def sample_nc2x(theta=0,df=100,size=1000):
    nc = theta * np.sqrt(2 * df)
    Y = chi2.rvs(df - 1, size=size)
    Z = norm.rvs(size=size)
    Y += nc + 2 * np.sqrt(nc) * Z + Z ** 2
    return Y

df = 10_000
theta_list = [0,0.1,0.25,0.5,1.,2.,4.]
samples = dict.fromkeys(theta_list)

#%%

for theta in theta_list:
    Y = sample_nc2x(theta=theta,df=df,size=1_000_000)
    samples[theta] = (Y - df)/np.sqrt(2*df)

#%%

for theta in samples.keys():
    # parameterise our distributions
    d1 = norm(theta, np.sqrt(1 + theta/np.sqrt(df/8)))
    # create new figure with size given explicitly
    plt.figure(figsize=(8, 5))
    y = samples[theta]
    # add histogram showing individual components
    plt.hist(y, 50, density=True, alpha=0.25, edgecolor='none')

    # get X limits and fix them
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)

    # add our distributions to figure
    x = np.linspace(mn, mx, 301)
    plt.plot(x, d1.pdf(x), color='C0',lw=2,label='theta = {:.2f}'.format(theta))
    plt.legend()
    plt.title('Non-central chi square')
    plt.ylabel('Density')
    filename = 'non-central-chi2_' + str(theta) + '.png'
    plt.savefig(filename)
    # plt.show()

#%%


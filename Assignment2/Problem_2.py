import numpy as np
import matplotlib.pyplot as plt

# Fixed parameters
sigma2 = 1
sigma_q2 = 1
x_val = 1
phi_values = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]

# True log-likelihood
def log_likelihood(theta, x, sigma2):
    var = theta**2 + sigma2
    return -0.5 * np.log(2 * np.pi * var) - (x**2) / (2 * var)

# ELBO function
def elbo(theta, phi, x, sigma2, sigma_q2):
    exp_log_likelihood = -0.5 * np.log(2*np.pi*sigma2) - 0.5/sigma2 * ((x - theta*phi)**2 + theta**2 * sigma_q2)
    kl = 0.5 * (phi**2 + sigma_q2 - 1 - np.log(sigma_q2))
    return exp_log_likelihood - kl

theta_vals = np.linspace(-10, 10, 500)
true_ll = np.array([log_likelihood(theta, x_val, sigma2) for theta in theta_vals])

plt.figure(figsize=(12, 8))
plt.plot(theta_vals, true_ll, 'k-', linewidth=3, label=r'True $\log P_\theta(x)$')


colors = plt.cm.viridis(np.linspace(0, 1, len(phi_values)))
for phi, color in zip(phi_values, colors):
    elbo_vals = np.array([elbo(theta, phi, x_val, sigma2, sigma_q2) for theta in theta_vals])
    plt.plot(theta_vals, elbo_vals, '--', color=color, linewidth=2, label=fr'ELBO ($\phi = {phi}$)')


plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$\log P_\theta(x)$', fontsize=14)
plt.title(r'True Log-Likelihood vs ELBO for Different $\phi$', fontsize=16)
plt.ylim(-5, 0)
plt.xlim(-10, 10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('./outputs/elbo_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
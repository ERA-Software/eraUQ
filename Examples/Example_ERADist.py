from Classes.ERADist import ERADist
from Classes.ERADist import Gaussian
from Classes.ERADist import Lognormal
from Classes.ERADist import Beta
from Classes.ERANataf import ERANataf
import numpy as np
import matplotlib.pylab as plt

'''
---------------------------------------------------------------------------
Example file: Definition and use of ERADist objects
---------------------------------------------------------------------------
 
 In this example a lognormal distribution is defined by its parameters,
 moments and data.Furthermore the different methods of ERADist are
 illustrated.
 For other distributions and more information on ERADist please have a look
 at the provided documentation or execute the command 'help(ERADist)'.
 
---------------------------------------------------------------------------
Developed by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Luca Sardi
Alexander von Ramm
Matthias Willer
Peter Kaplan

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2021-03
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
'''

np.random.seed(2026) #initializing random number generator

m1 = 0
std1 = 1

m2 = 2
std2 = 1

Gaussian_abstract_1 = Gaussian("MOM", [m1, std1])
Gaussian_abstract_2 = Gaussian("PAR", [m2, std2], ID="Identitiy_to_function")

print(Gaussian_abstract_1.ID)
print(Gaussian_abstract_2.ID)

print(Gaussian_abstract_1.mean())

print("Verifying the Lognormal one:")

par_LN = [0, 1]

mom_LN = [np.exp(par_LN[0] + 0.5 * par_LN[1]**2), np.sqrt((np.exp(par_LN[1]**2) - 1) * np.exp(2*par_LN[0] + par_LN[1]**2)) ]

lognormal_abstract_1 = Lognormal("MOM", mom_LN, ID="Moment Defined")
lognormal_abstract_2 = Lognormal("PAR", par_LN, ID="Parameter Defined")

print(lognormal_abstract_1.mean())
print(lognormal_abstract_2.mean())

print(lognormal_abstract_1.std())
print(lognormal_abstract_2.std())

print("Verifying Beta one:")

r = 0.5
s = 0.5
a = 0
b = 1

analytical_mean_beta = r / (r + s)
beta_abstract_1 = Beta("PAR", [r, s, a, b])

ns = 5000
beta_samples = beta_abstract_1.random(ns)

# analytical grid
x = np.linspace(a, b, 1000)
pdf_value = beta_abstract_1.pdf_function(x)

# plot
plt.hist(beta_samples, bins=50, density=True, alpha=0.5)
plt.plot(x, pdf_value, linewidth=2)

plt.xlabel("x")
plt.ylabel("El PDF-o")
plt.title("Beta distribution: PDF + samples")
plt.show()


# OHA?! NANIII?!!
# A Nataf?
print("Nataf Timeeee")

RVector = ERANataf([Gaussian_abstract_1, lognormal_abstract_1], [ [1, 0], [0 , 1] ])

ns = 1000
samples = RVector.random(ns)

# sample mean
mean = samples.mean(axis=0)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# --- Scatter plot (bottom-left) ---
axs[1, 0].scatter(samples[:, 0], samples[:, 1])
axs[1, 0].scatter(mean[0], mean[1], marker='x', s=100)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
axs[1, 0].set_title("Scatter plot")

# --- Histogram of x (top-left) ---
axs[0, 0].hist(samples[:, 0], bins=30)
axs[0, 0].set_title("Histogram of x")

# --- Histogram of y (bottom-right) ---
axs[1, 1].hist(samples[:, 1], bins=30)
axs[1, 1].set_title("Histogram of y")

# --- Empty top-right subplot ---
axs[0, 1].axis("off")

plt.show()


print("End of Code :C")
''' Definition of an ERADist object by the distribution parameters '''
"""
dist = ERADist('lognormal','PAR',[2,0.5])

# computation of the first two moments
mean_dist = dist.mean()
std_dist = dist.std()

# generation of n random samples
n = 10000
samples = dist.random(n)

''' Definition of an ERADist object by the first moments
 Based on the just determined moments a new distribution object with the
 same properties is created... '''

dist_mom = ERADist('lognormal','MOM',[mean_dist,std_dist])

''' Definition of an ERADist object by data fitting
 Using maximum likelihood estimation a new distribution object is created
 from the samples which were created above.'''

dist_data = ERADist('lognormal','DATA',samples)


''' Other methods '''

# generation of n samples x to work with
x = dist.random(n)

# computation of the PDF for the samples x
pdf = dist.pdf(x)

# computation of the CDF for the samples x
cdf = dist.cdf(x)

# computation of the inverse CDF based on the CDF values (-> initial x)
icdf = dist.icdf(cdf)


''' Plot of the PDF and CDF '''

x_plot = np.linspace(0,40,200);     # values for which the PDF and CDF are evaluated 
pdf = dist.pdf(x_plot);     # computation of PDF
cdf = dist.cdf(x_plot);     # computation of CDF

fig_dist = plt.figure(figsize=[16, 9])

fig_pdf = fig_dist.add_subplot(121)
fig_pdf.plot(x_plot, pdf)
fig_pdf.set_xlabel(r'$X$')
fig_pdf.set_ylabel(r'$PDF$')

fig_cdf = fig_dist.add_subplot(122)
fig_cdf.plot(x_plot, cdf)
fig_cdf.set_xlabel(r'$X$')
fig_cdf.set_ylabel(r'$CDF$')

"""
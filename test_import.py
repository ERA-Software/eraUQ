from eradist import ERADist
import matplotlib.pyplot as plt

# Copy this file into a different directory, in order to test the installed local package. I say different directory,
# because that way you know that the functions are working not because they are in the same directory, but because
# the package has been added to the package list used by your python interpreter...

dist = ERADist("Normal","MOM",[0,4])
ns = 10000
xs = dist.random(ns)

plt.hist(xs,50)
plt.show()

print("End of Code :C")
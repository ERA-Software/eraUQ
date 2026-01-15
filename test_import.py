from eradist import ERADist
import matplotlib.pyplot as plt

dist = ERADist("Normal","MOM",[0,4])
ns = 10000
xs = dist.random(ns)

plt.hist(xs,50)
plt.show()

print("End of Code :C")
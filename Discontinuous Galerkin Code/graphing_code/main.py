import numpy as np
import matplotlib.pyplot as plt
import math


file = np.loadtxt("DGout.txt")




exact = file[9]
x = np.linspace(0, 40, 80)

plt.plot(x, exact, label="exact")
plt.xlabel("Grid Point")
plt.ylabel("U(x,t)")
plt.title("U at time t=5s for DG Scheme")
plt.legend()
plt.savefig(str(points[i]) + "graph.png")
plt.show()

"""# This all graphs the error:

errorData = np.loadtxt("errors.txt")

for i in range(len(errorData)):
    for j in range(5):
        errorData[i][j] = math.log(errorData[i][j])

numPoints = [math.log(40), math.log(80), math.log(160), math.log(320), math.log(1280)]

plt.plot(numPoints, errorData[0][0:5], label="Upwind")
plt.plot(numPoints, errorData[1][0:5], label="Lax")
plt.plot(numPoints, errorData[2][0:5], label="Lax-Wendroff")
plt.plot(numPoints, errorData[3][0:5], label="Leap-Frog")
plt.plot(numPoints, errorData[4][0:5], label="MacCormack")
plt.xlabel("Log(# of Grid Points)")
plt.ylabel("log(E)")
plt.title("Error for Various Schemes on the Linear Advection Equation")
plt.legend()
plt.savefig("errorGraph.png")
plt.show()"""
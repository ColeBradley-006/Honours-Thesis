import numpy as np
import matplotlib.pyplot as plt
import math

filenames = [["exact0.txt", "lax0.txt", "laxWendroff0.txt", "LeapFrog0.txt", "maccormack0.txt", "upwind0.txt"],
             ["exact1.txt", "lax1.txt", "laxWendroff1.txt", "LeapFrog1.txt", "maccormack1.txt", "upwind1.txt"],
             ["exact2.txt", "lax2.txt", "laxWendroff2.txt", "LeapFrog2.txt", "maccormack2.txt", "upwind2.txt"],
             ["exact3.txt", "lax3.txt", "laxWendroff3.txt", "LeapFrog3.txt", "maccormack3.txt", "upwind3.txt"]]

points = [10,20,41,82]
datasets = []
for i in range(4):
    cur = []
    for each in filenames[i]:
        cur.append(np.loadtxt(each))
    datasets.append(cur)

#This all graphs the different schemes
for i in range(4):

    
    exact = datasets[i][0][10]
    lax = datasets[i][1][10]
    laxW = datasets[i][2][10]
    lf = datasets[i][3][10]
    mac = datasets[i][4][10]
    up = datasets[i][5][10]
    x = np.linspace(0,40,points[i])

    plt.plot(x,exact, label = "exact")
    plt.plot(x,lax, label = "Lax")
    plt.plot(x,laxW, label = "Lax-Wendroff")
    plt.plot(x,lf, label = "Leap-Frog")
    plt.plot(x,mac, label = "MacCormack")
    plt.plot(x,up, label = "Upwind")
    plt.xlabel("Grid Point")
    plt.ylabel("U(x,t)")
    plt.title("U at time t=5s for " + str(points[i]) + " grid points")
    plt.legend()
    plt.savefig(str(points[i]) +"graph.png")
    plt.show()

#This all graphs the error:

errorData = np.loadtxt("errors.txt")

for i in range(len(errorData)):
    for j in range(4):
        errorData[i][j] = math.log(errorData[i][j])

numPoints = [math.log(10),math.log(20),math.log(41),math.log(82)]

plt.plot(numPoints, errorData[0][0:4], label = "Upwind")
plt.plot(numPoints, errorData[1][0:4], label = "Lax")
plt.plot(numPoints, errorData[2][0:4], label = "Lax-Wendroff")
plt.plot(numPoints, errorData[3][0:4], label = "Leap-Frog")
plt.plot(numPoints, errorData[4][0:4], label = "MacCormack")
plt.xlabel("Log(# of Grid Points)")
plt.ylabel("log(E)")
plt.title("Error for Various Schemes on the Linear Advection Equation")
plt.legend()
plt.savefig("errorGraph.png")
plt.show()
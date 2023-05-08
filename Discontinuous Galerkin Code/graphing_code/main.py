import numpy as np
import matplotlib.pyplot as plt
import math

filenames = ["0DGout.txt", "1DGout.txt", "2DGout.txt", "3DGout.txt"]

points = [10, 20, 40, 80]
cur = []
for each in filenames:
    cur.append(np.loadtxt(each))
file = np.loadtxt("DGSine5s.txt")

exact1 = np.linspace(0, 2*np.pi, 10)
exact1 = np.sin(exact1-0.5*5) + 2
exact2 = np.linspace(0, 2*np.pi, 20)
exact2 = np.sin(exact2-0.5*5) + 2
exact3 = np.linspace(0, 2*np.pi, 40)
exact3 = np.sin(exact3-0.5*5) + 2
exact4 = np.linspace(0, 2*np.pi, 80)
exact4 = np.sin(exact4-0.5*5) + 2

y1 =  cur[0][8]
y2 =  cur[1][8]
y3 =  cur[2][8]
y4 =  cur[3][8]
x1 = np.linspace(0, 2*math.pi, 10)
x2 = np.linspace(0, 2*math.pi, 20)
x3 = np.linspace(0, 2*math.pi, 40)
x4 = np.linspace(0, 2*math.pi, 80)



plt.plot(x1, y1, label="10 Elements")
plt.plot(x2, y2, label="20 Elements")
plt.plot(x3, y3, label="30 Elements")
plt.plot(x4, y4, label="40 Elements")
plt.plot(x4, exact4, label="Exact Solution")
plt.xlabel("x")
plt.ylabel("U(x,t)")
plt.title("Advecting a Sine wave after 5s for DG Scheme")
plt.legend()
plt.savefig("DGgraph.png")
plt.show()

y1error= exact1 - y1
y2error= exact2 - y2
y3error= exact3 - y3
y4error= exact4 - y4

S1 = np.sum(np.square(y1error))
S2 = np.sum(np.square(y2error))
S3 = np.sum(np.square(y3error))
S4 = np.sum(np.square(y4error))

errorPoints = np.array([math.log(math.sqrt(S1*2*math.pi/10)), math.log(math.sqrt(S2*2*math.pi/20)), math.log(math.sqrt(S3*2*math.pi/40)), math.log(math.sqrt(S4*2*math.pi/40))])
errorX = np.array([math.log(10), math.log(20), math.log(40), math.log(80)])
plt.plot(errorX, errorPoints)
plt.xlabel("log(# of elements)")
plt.ylabel("log(E)")
plt.title("Error of DG scheme for varying # of elements")
plt.savefig("errorGraph.png")
plt.show()

order = math.log((math.sqrt(S4*2*math.pi/40)-math.sqrt(S3*2*math.pi/40))/(math.sqrt(S3*2*math.pi/40)-math.sqrt(S2*2*math.pi/20)))/(math.log(2))
print(order)
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
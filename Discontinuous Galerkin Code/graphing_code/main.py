import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import lagrange


"""NEW CODE"""

"""This returns a function which then allows for an x-value to be input to solve for the points"""



filenames = [["P1_E0_DGout.txt", "P1_E1_DGout.txt", "P1_E2_DGout.txt", "P1_E3_DGout.txt", "P1_E4_DGout.txt", "P1_E5_DGout.txt"],
             ["P2_E0_DGout.txt", "P2_E1_DGout.txt", "P2_E2_DGout.txt", "P2_E3_DGout.txt", "P2_E4_DGout.txt", "P2_E5_DGout.txt"],
             ["P3_E0_DGout.txt", "P3_E1_DGout.txt", "P3_E2_DGout.txt", "P3_E3_DGout.txt", "P3_E4_DGout.txt", "P3_E5_DGout.txt"],
             ["P4_E0_DGout.txt", "P4_E1_DGout.txt", "P4_E2_DGout.txt", "P4_E3_DGout.txt", "P4_E4_DGout.txt", "P4_E5_DGout.txt"],
             ["P5_E0_DGout.txt", "P5_E1_DGout.txt", "P5_E2_DGout.txt", "P5_E3_DGout.txt", "P5_E4_DGout.txt", "P5_E5_DGout.txt"]]
filenamesx = [["X_P1_E0_DGout.txt", "X_P1_E1_DGout.txt", "X_P1_E2_DGout.txt", "X_P1_E3_DGout.txt", "X_P1_E4_DGout.txt", "X_P1_E5_DGout.txt"],
             ["X_P2_E0_DGout.txt", "X_P2_E1_DGout.txt", "X_P2_E2_DGout.txt", "X_P2_E3_DGout.txt", "X_P2_E4_DGout.txt", "X_P2_E5_DGout.txt"],
             ["X_P3_E0_DGout.txt", "X_P3_E1_DGout.txt", "X_P3_E2_DGout.txt", "X_P3_E3_DGout.txt", "X_P3_E4_DGout.txt", "X_P3_E5_DGout.txt"],
             ["X_P4_E0_DGout.txt", "X_P4_E1_DGout.txt", "X_P4_E2_DGout.txt", "X_P4_E3_DGout.txt", "X_P4_E4_DGout.txt", "X_P4_E5_DGout.txt"],
             ["X_P5_E0_DGout.txt", "X_P5_E1_DGout.txt", "X_P5_E2_DGout.txt", "X_P5_E3_DGout.txt", "X_P5_E4_DGout.txt", "X_P5_E5_DGout.txt"]]


points = {0:3,1:6,2:12,3:24,4:48,5:96}
cur = []
for each in filenames:
    new = []
    for ele in each:
        new.append(np.loadtxt(ele))
    cur.append(new)

curx = []
for each in filenamesx:
    new = []
    for ele in each:
        new.append(np.loadtxt(ele))
    curx.append(new)

storage = np.zeros(shape=(5,6))

for count, p in enumerate(cur):
    for incount, data in enumerate(p):
        xpoints = curx[count][incount]
        ypoints = cur[count][incount]
        outputy = np.array([])
        outputx = np.array([])
        for i in range(points[incount]):
            poly = lagrange(xpoints[:,i], ypoints[:,i])
            x = np.linspace(xpoints[0,i], xpoints[count + 1, i], 100)
            y = poly(x)
            outputx = np.concatenate((outputx,x))
            outputy = np.concatenate((outputy,y))

        number = outputx.size
        exact = np.sin(outputx-0.5*5) + 2
        plt.plot(outputx, outputy, label="P" + str(count + 1) + " " + str(points[incount]) + " Elements")
        error = exact - outputy
        S = np.sum(np.square(error))
        trueError = math.sqrt(S * 2.0 * math.pi / number)
        storage[count][incount] = trueError
        if incount == 4:
            plt.plot(outputx, exact, label="Exact Solution")
    plt.legend()
    plt.savefig(str(count) + "errorGraph.png")
    plt.show()


errorX = np.array([math.log(3), math.log(6), math.log(12), math.log(24), math.log(48), math.log(96)])
for count, bar in enumerate(storage):
        y = np.log(bar)
        plt.plot(errorX, y, label="P" + str(count + 1) )
        if count > 2:
            order = abs(math.log((bar[2]-bar[1]) / (bar[1]-bar[0]))/math.log(2))
            print(order)
        elif count == 2:
            order = abs(math.log((bar[3]-bar[2])/(bar[2]-bar[1])) / math.log(2))
            print(order)
        elif count == 1:
            order = abs(math.log((bar[4]-bar[3])/(bar[3]-bar[2])) / math.log(2))
            print(order)
        else:
            order = abs(math.log((bar[4]-bar[3])/(bar[3]-bar[2])) / math.log(2))
            print(order)
plt.xlabel("log(# of elements)")
plt.ylabel("log(E)")
plt.title("Error of DG scheme for varying # of elements")
plt.legend()
plt.savefig("errorGraph.png")
plt.show()


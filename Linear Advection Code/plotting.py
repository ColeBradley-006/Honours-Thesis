import numpy as np
import matplotlib.pyplot as plt

filenames = ["exact.txt", "lax.txt", "laxWendroff.txt", "LeapFrog.txt", "maccormack.txt", "upwind.txt"]

datasets = []
for each in filenames:
    datasets.append(np.loadtxt(each))

for i in range(21):

    
    exact = datasets[0][i]
    lax = datasets[1][i]
    laxW = datasets[2][i]
    lf = datasets[3][i]
    mac = datasets[4][i]
    up = datasets[5][i]
    x = np.linspace(0,40,41)

    plt.plot(x,exact, label = "exact")
    plt.plot(x,lax, label = "Lax")
    plt.plot(x,laxW, label = "Lax-Wendroff")
    plt.plot(x,lf, label = "Leap-Frog")
    plt.plot(x,mac, label = "MacCormack")
    plt.plot(x,up, label = "Upwind")
    plt.title("U at time t=" + str(i / 2.0) + "s")
    plt.legend()
    plt.show()

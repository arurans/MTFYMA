from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from deCasteljau import *

def animate_bezier(frames, interval, P, t, entire = False):
    """
    Animate construction of Bézier curve

    Arguments
    ---------------------------------------
        frames: int
            Number of frames in the animation
        interval: int
            number of milliseconds between each frame
        P: 2 x n - matrix
            dim is the dimension of the points and n is number of points
        t: numpy-array
            Array of time-steps
        entire: bool (default: false)
            Animate for the whole time if true, else animate only up to t = 1/3
    
    Returns
    ---------------------------------------
    anim: the gif
    """

    fig = plt.figure()
    ax = plt.axes(xlim = (0,8), ylim = (-3.5, 3.5))

    title = ax.text(0.1, 3.2, f"t = {0}")

    plt.grid()
    plt.title("Animated Bézier curve")
    plt.xlabel("x")
    plt.ylabel("y")

    #Plot the fixed linear line
    for i in range(4):
        linear = np.array([deCasteljau(P, t)[1][1,:,i] for t in t])
        plt.plot(linear[:,0], linear[:,1], color = "red")
        plt.scatter(linear[:,0][0], linear[:,1][0], color = "red")

    plt.scatter(linear[:,0][-1], linear[:,1][-1], color = "red")

    #The bezier curve
    bezierx = []
    beziery = []
    bezier, = ax.plot([], [], color = "black", lw = 3)

    #The linear part
    linx = [0, 0, 0, 0]
    liny = [0, 0, 0, 0]
    lin, = ax.plot([], [], "bo-", lw = 3)

    #The quadratic part
    quadx = [0, 0, 0]
    quady = [0, 0, 0]
    quad, = ax.plot([], [], "go-", lw = 3)

    #The cubic part
    cubx = [0, 0]
    cuby = [0, 0]
    cub, = ax.plot([], [], "mo-", lw = 3)

    point, = ax.plot(0, 1, color = "black", marker = "o", lw = 5)
    
    def draw_bezier(i):
        if entire:#Animate the entire bezier curve
            if i <= 8*frames//9:
                t = 9/(8 * frames) * i
            else:
                t = 1
        else:#Animate only up to t = 1/3
            if i <= 2*frames//3:
                t = 1/(2 * frames) * i
            else:
                t = 1/3

        p, Pvecs = deCasteljau(P, t)
        degree, dim, n = Pvecs.shape

        for m in range(n - 1):
            linx[m] = Pvecs[1, 0, m]
            liny[m] = Pvecs[1, 1, m]
        
        for m in range(n - 2):
            quadx[m] = Pvecs[2, 0, m]
            quady[m] = Pvecs[2, 1, m]
        
        for m in range(n - 3):
            cubx[m] = Pvecs[3, 0, m]
            cuby[m] = Pvecs[3, 1, m]


        bezierx.append(p[0])
        beziery.append(p[1])

        bezier.set_data(bezierx, beziery)
        lin.set_data(linx, liny)
        quad.set_data(quadx, quady)
        cub.set_data(cubx, cuby)

        point.set_data([p[0]], [p[1]])

        title.set_text(f"t = {t:.3f}")

        return [bezier, lin, quad, cub, point]

    anim = FuncAnimation(fig, draw_bezier, frames = frames, interval = interval, blit = True)

    return anim
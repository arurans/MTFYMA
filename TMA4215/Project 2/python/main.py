import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from matplotlib.animation import FuncAnimation
import animation

from plots import *
from compositeBezier import *

plt.rcParams["figure.figsize"] = (16,10)#Increase figure size

#Points to interpolat
P = np.array([
    [0, 2, 4, 6, 8],
    [0, 3, 0, -3, 0]
])

#The letter "S"
A=np.array([
[2.5,3.25,3.5,2.75,1.75,2.5,3.5,2.5,1.5,1.25,2.4,3.0,2.5,1.25],
[0.0,0.5,1.5,3.25,4.75,5.4,5.0,6,5.6,4.5,2.75,1.5,0.7,1.5]
])

V=np.array([
[1.0,1.0,0.0,-1.0,0.0,1.0,1.0,-1.0,-1.0,0.0,1.0,-0.1,-1.0,-1.0],
[0.0,1.0,1.0,1.0,1.0,0.0,1.0,0.0,-1.0,-1.0,-1.0,-1.0,0.0,-1.0]
])

#Rotate the letter pi/4 counter-clockwise
R = np.array([
    [np.cos(np.pi/4), -np.sin(np.pi/4)],
    [np.sin(np.pi/4), np.cos(np.pi/4)]
])

#Displace and mirror letter
S_matrix = np.array([
    [1, 1/2],
    [1, 0]
])

S_const = np.array([1/2, 1/2])

#rotate and displace letter
T_matrix = np.array([
    [-1, -1/2],
    [1, 0]
])

T_const = np.array([1/2, -1/2])

#Time-values
t = np.linspace(0, 1, 1000)

def main():
    #plot_Bernstein(3, t)
    #plot_Bernstein(9, t)
    #anim = animation.animate_bezier(300, 20, P, t, entire = True)      #UNCOMMENT TO RENDER ANIMATION
    #anim.save('bézier.gif', writer='imagemagick') #Save gif to a file  #UNCOMMENT TO SAVE ANIMATION
    P_1, t_1 = interpolate_periodic(A, V)
    draw_bezier(2*P_1, t_1)
    # draw_bezier(mod_P(P_1, R), t_1)
    # draw_bezier(mod_P(P_1, S_matrix, S_const), t_1)
    # draw_bezier(mod_P(P_1, T_matrix, T_const), t_1)

if __name__ == "__main__":
    main()

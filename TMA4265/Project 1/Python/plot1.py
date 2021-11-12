from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (15, 7) #Increase figure size

def plotCommands(xlabel, ylabel, title, path, xscale = False, yscale = False, legend = False, printString = ""):
    #Function containing the usual commands when plotting with matplotlib.pyplot
    #If log-scaled x- and/or y-axis is needed, set xscale and/or yscale to True
    #If the figure has a label, set legend = True
    #If extra information is needed, use printString
    plt.xlabel(xlabel, size = 20)
    plt.ylabel(ylabel, size = 20)
    plt.title(title, size = 27)
    if xscale:
        plt.xscale('log')
    if yscale:
        plt.yscale('log')
    if legend:
        plt.legend()
    plt.grid()
    if printString:
        print(printString)
    plt.savefig(path)
    return None


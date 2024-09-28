# make a class of plot contain general settings (axis, font, legend, etc)

# output: ax, fig handles
import matplotlib.pyplot as plt
import os
import matplotlib

class StartPlots:
    # set figure size
    plt.rcParams['figure.figsize'] = [10, 10]
    # set font
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20

    # set line width
    plt.rcParams["lines.linewidth"] = 3

    # set line width of x/y axis
    plt.rcParams["axes.linewidth"] = 2

    # set colormap
    plt.rcParams['image.cmap'] = 'jet'
    matplotlib.use('Qt5Agg')

    def __init__(self):

        self.fig, self.ax = plt.subplots(layout="constrained")

        # set plot top/right boundaries to invisible
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

    def legend(self, leg):
        # add legend
        legend = self.ax.legend(leg)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_facecolor('none')

        return legend

    def save_plot(self, title, type, figpath):
        # save fig in specified type.
        # 'tiff', 'svg', etc
        """
        :param title: name of the figure, '*.pdf'
        :param type: extension name of the figure, 'tiff', 'svg', etc.
        :param figpath: the desired directory to save the figure
        :return:
        """
        # check if the directory exist
        if not os.path.exists(figpath):
            os.makedirs(figpath)

        self.fig.savefig(os.path.join(figpath,title), format=type)

# another class to generate subplots
class StartSubplots(StartPlots):

    def __init__(self, xdim, ydim, ifSharex=False, ifSharey=False):
        self.fig, self.ax = plt.subplots(xdim, ydim,  layout="constrained", sharex=ifSharex, sharey=ifSharey)
        #plt.tight_layout()

        # set plot top/right boundaries to invisible
        if xdim==1 and ydim > 1:
            for yy in range(ydim):
                self.ax[yy].spines['top'].set_visible(False)
                self.ax[yy].spines['right'].set_visible(False)
        elif xdim > 1 and ydim == 1:
            for xx in range(xdim):
                self.ax[xx].spines['top'].set_visible(False)
                self.ax[xx].spines['right'].set_visible(False)
        else:
            for xx in range(xdim):
                for yy in range(ydim):
                    self.ax[xx,yy].spines['top'].set_visible(False)
                    self.ax[xx,yy].spines['right'].set_visible(False)


# test
import numpy as np

if __name__ == "__main__":
    Fig1 = StartPlots()

    x = np.arange(0,np.pi,0.01)
    y = np.sin(x)

    Fig1.ax.plot(x,y)
    plt.show()

    savePath = "C:\\Users\\hongl\\Desktop\\tesetfig"

    Fig1.save_plot('testFig1.pdf', 'pdf', savePath)
    Fig2 = StartSubplots(2,2)

    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    y4 = x

    Fig2.ax[0,0].plot(x,y1)
    Fig2.ax[0,1].plot(x, y2)
    Fig2.ax[1,0].plot(x,y3)
    Fig2.ax[1,1].plot(x,y4)
    plt.show()

    Fig2.save_plot('testFig2.tiff', 'tiff', savePath)
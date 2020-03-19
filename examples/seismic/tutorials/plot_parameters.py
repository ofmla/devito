import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import matplotlib.ticker as plticker

def plot(v,epsilon,delta,theta):

 fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
 fig.subplots_adjust(wspace=0.25)
 fig.subplots_adjust(hspace=0.25)

 im1=ax1.imshow(v.T, cmap=plt.cm.jet)
 ticks = ax1.get_yticks()*0.01
 loc = plticker.MultipleLocator(base=100.0)
 ax1.xaxis.set_major_locator(loc)
 ax1.set_yticklabels(ticks)
 ax1_divider = make_axes_locatable(ax1)
 cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
 cb1 = plt.colorbar(im1, cax=cax1)
 cb1.ax.tick_params(labelsize=10)
#cb1.set_label('v (km/s)')
 cb1.ax.set_title('v (km/s)',size=12)

 im2=ax2.imshow(epsilon.T, cmap=plt.cm.hot) 
 loc = plticker.MultipleLocator(base=100.0)
 ax2.xaxis.set_major_locator(loc)
 ax2_divider = make_axes_locatable(ax2)
 cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")
 cb2 = plt.colorbar(im2, cax=cax2)
 cb2.ax.tick_params(labelsize=10)
#cb2.set_label('ε')
 cb2.ax.set_title('ε',size=12)

 im3=ax3.imshow(delta.T, cmap=plt.cm.hot) 
 ticks = ax3.get_xticks()*0.01
 loc = plticker.MultipleLocator(base=100.0)
 ax3.xaxis.set_major_locator(loc)
 ax3.set_yticklabels(ticks)
 ax3.set_xticklabels(ticks)
 ax3_divider = make_axes_locatable(ax3)
 cax3 = ax3_divider.append_axes("right", size="7%", pad="2%")
 cb3 = plt.colorbar(im3, cax=cax3,format='%.1e')
 cb3.ax.tick_params(labelsize=10)
#cb3.set_label('δ')
 cb3.ax.set_title('δ',size=12)

 im4=ax4.imshow(theta.T, cmap=plt.cm.jet) 
 ticks = ax4.get_xticks()*0.01
 loc = plticker.MultipleLocator(base=100.0)
 ax4.xaxis.set_major_locator(loc)
 ax4.set_xticklabels(ticks)
 ax4_divider = make_axes_locatable(ax4)
 cax4 = ax4_divider.append_axes("right", size="7%", pad="2%")
 cb4 = plt.colorbar(im4, cax=cax4)
 cb4.ax.tick_params(labelsize=10)
#cb4.set_label('θ')
 cb4.ax.set_title('θ',size=12)

 for ax in (ax1,ax2,ax3,ax4):
    ax.set(xlabel='Distance (km)', ylabel='Depth (km)')
 for ax in (ax1,ax2,ax3,ax4):
    ax.label_outer()

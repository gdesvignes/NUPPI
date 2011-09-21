import sys, gtk, gobject, time
import matplotlib
matplotlib.use('GTKAgg')
from nuppi_utils import Status, Databuf
import shm_wrapper as shm
import numpy as n
import pylab as p
from nuppi_utils import *

nspec_sum = 512
poln  = 0
polns = 'IQUV'
delay = 0.1
view_mean = True
view_dB = False

def dB(x, mindb=-80.0):
    return n.where(x>0, 10.0*n.log10(x/255.0), mindb)

def usage():
    print """
  GUPPI Real-Time Monitor
  -----------------------
  This program provides a simple real-time monitor for GUPPI data.
  There are several options to adjust plotting.  Position the mouse
  in the plotting window and press one of the following keys:
    '1-4':     Change the poln number to plot (1='I', 2='Q', etc)
    'i', 'o':  Zoom in/out of the X-axis
    '-', '+':  Zoom in/out of the Y-axis (no shift needed)
    'u', 'd':  Shift plot up/down in the window
    '<', '>':  Shift plot left/right in the window (no shift needed)
    'x', 'c':  Change X-axis units to/from frequency from/to channels
    'b':       Change Y-axis units to/from dB from/to linear scale
    'a':       Attempt to auto-scale the plot
    's':       Toggle between the mean and standard deviation
    'q':       Quit the plotter
  You can also use the icons at the bottom of the window to scale, 
  shift, zoom, print etc.
    """

def key_press(event):
    global poln, view_mean, view_dB
    if event.key in 'io':    # Zoom in/out (x-direction)
        axes = main_line.get_axes()
        xlim = axes.get_xlim()
        midx = 0.5*(xlim[1]+xlim[0])
        dx = 0.5*(xlim[1]-xlim[0])
        if event.key=='i':  # Zoom in  by a factor of 2
            axes.set_xlim(midx-0.5*dx, midx+0.5*dx)
        else:               # Zoom out by a factor of 2
            axes.set_xlim(midx-2.0*dx, midx+2.0*dx)
    elif event.key in '=-': # Zoom in/out (y-direction)
        axes = main_line.get_axes()
        ylim = axes.get_ylim()
        midy = 0.5*(ylim[1]+ylim[0])
        dy = 0.5*(ylim[1]-ylim[0])
        if event.key=='=':  # Zoom in  by a factor of 2
            axes.set_ylim(midy-0.5*dy, midy+0.5*dy)
        else:               # Zoom out by a factor of 2
            axes.set_ylim(midy-2.0*dy, midy+2.0*dy)
    elif event.key in 'ud':  # Shift up/down by 20% of the y-height
        axes = main_line.get_axes()
        ylim = axes.get_ylim()
        dy = ylim[1]-ylim[0]
        if event.key=='d':   # Shift down
            axes.set_ylim(ylim+0.2*dy)
        else:                # Shift up
            axes.set_ylim(ylim-0.2*dy)
    elif event.key in ',.':    # Shift right/left by 20% of the x-width
        axes = main_line.get_axes()
        xlim = axes.get_xlim()
        dx = xlim[1]-xlim[0]
        if event.key=='.':     # Shift right
            axes.set_xlim(xlim+0.2*dx)
        else:                  # Shift left
            axes.set_xlim(xlim-0.2*dx)
    elif event.key in 'xc':  # Freq to bins and vise-versa
        if ax.get_xlabel().startswith('Chan'):
            ax.set_xlabel('Frequency (MHz)')
            main_line.set_xdata(freqs)
            min_line.set_xdata(freqs)
            max_line.set_xdata(freqs)
            axes = main_line.get_axes()
            axes.set_xlim(freqs[0], freqs[-1])
        else:
            ax.set_xlabel('Channel Number')
            main_line.set_xdata(bins)
            min_line.set_xdata(bins)
            max_line.set_xdata(bins)
            axes = main_line.get_axes()
            axes.set_xlim(bins[0], bins[-1])
    #elif event.key=='l':  # Log axes
    #    axes = main_line.get_axes()
    #    axes.set_yscale('log')
    #    axes.autoscale_view()
    elif event.key=='a':  # Auto-scale
        axes = main_line.get_axes()
        axes.autoscale_view()
    elif event.key=='b': # dB scale
        view_dB = not view_dB
        axes = main_line.get_axes()
        ylim = axes.get_ylim()
        if view_dB:
            print "Changing power units to dB"
            axes.set_ylim(-50.0, 0.0)
            ax.set_ylabel('Arbitrary Power (dB)')
        else:
            print "Changing power units to a linear scale"
            axes.set_ylim(-255.0, 255)
            ax.set_ylabel('Arbitrary Power')
    elif event.key in '1234': # Switch polarization
        poln = int(event.key)-1
        ax.set_title("Poln is '%s'"%polns[poln])
    elif event.key=='s':  # Stats
        view_mean = not view_mean
        if view_mean: print "Main plot is average spectra"
        else: print "Main plot is spectra standard deviation"
    elif event.key=='q':  # Quit
        sys.exit()
    fig.canvas.draw()

def update_lines(*args):
    g.read()
    try:
        curblock = g["CURBLOCK"]
    except KeyError:
        curblock = 1
    data = d.data(curblock)

    y = data[chan, :nspec, 0]
    main_line.set_ydata (data[chan, :nspec, 0])


    #idx = abs(main_spec).argmax()
    #print "Block %2d, poln '%s': Max chan=%d freq=%.3fMHz value=%.3f" %\
    #      (curblock, polns[poln], idx, freqs[idx], main_spec[idx]) 
    #ax.set_title("Chan #%d Poln is '%s'  Std=%f"%(chan, polns[poln], rms) )
    print "Block %2d" % (curblock) 

    fig.canvas.restore_region(background)
    ax.draw_artist(main_line)
    fig.canvas.blit(ax.bbox)
    #time.sleep(delay)

    

    return True

# Print the usage
usage()

# Set Logging
log = logfile("databuf_monitor.py")

# Get all of the useful values
g = Status(log)
g.read()
nchan = g["OBSNCHAN"]
totnchan = g["TOTNCHAN"]
npoln = g["NPOL"]
BW = g["OBSBW"]
fctr = g["OBSFREQ"]
blocsize = g["BLOCSIZE"]

# Sum different amount depending on nchan
#nspec_sum = 512 * 2048 / nchan

# Access the data buffer
d = Databuf()

dt = 1/ ((BW / totnchan)* 1e6)

data = d.data(0) # argument is the block in the data buffer
d.dtype = n.uint8

print 'd.blocsize (allocated) = %d,  blocksize used = %d'%(d.block_size, blocsize)
print "BW = %f, totnchan = %d"%(BW, totnchan)
print 'dt = %e'%dt
print 'len(data) = %d,  poln = %d\n'%(len(data), poln)

#n.set_printoptions(threshold=n.nan)

#print data[:1024*1024, poln, 0]
#print data[:nspec,poln]


# Initialize the plot
fig = p.figure()
fig.canvas.mpl_connect('key_press_event', key_press)
ax = fig.add_subplot(111) # #vert, #horiz, winnum


# Convert samples into time axis
#y = data[chan, :nspec, 0]
y = data
x = n.arange(0, len(data), 1)
#x = x*dt

#rms = n.std(y)

# Set title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Arbitrary Power')
#ax.set_title("Chan #%d Poln is '%s'  Std=%f"%(chan, polns[poln], rms) )
# save the clean slate background -- everything but the animated line
# is drawn and saved in the pixel buffer background
background = fig.canvas.copy_from_bbox(ax.bbox)


#print "Len",len(x), len (y)

main_line, = ax.plot(x, y, animated=True)

fig.canvas.draw()

# Start the event loop
gobject.timeout_add(1000,update_lines)
try:
    p.show()
except KeyboardInterrupt:
    print "Exiting.."

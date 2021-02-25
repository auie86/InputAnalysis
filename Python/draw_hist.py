# File:  draw_hist.py - Draws a histogram from the data in specified file
#
# Created (Jeff Smith) March 22, 2017
#
# $Id: draw_hist.py 1180 2017-03-21 14:50:58Z smitjef $
#-------------------------

# imports
import sys
import matplotlib.pyplot as plt

NumBins = 35
FileName = "data.txt"
if len(sys.argv) > 1:
    FileName = sys.argv[1]
    if len(sys.argv) > 2:
        NumBins = int(sys.argv[2])
vals = [float(i.rstrip()) for i in open(FileName,'r') if i.rstrip()]
plt.hist(vals, bins=NumBins, normed=False)
plt.axvline(0, color='r', linestyle='solid', linewidth=2)
plt.show()

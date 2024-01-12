"""
Adapted from pyapprox repo: https://github.com/sandialabs/pyapprox/blob/master/pyapprox/util/configure_plots.py
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
mystyle = {
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",   
        "text.usetex": True,            # use latex for all text handling            
        "font.family": "serif"
        }
mpl.rcParams.update(mystyle)
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf' # gives best resolution plots
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 16
#print mpl.rcParams.keys()
#mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}\usepackage{amsmath}\usepackage{amssymb}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}' # siunitx is not available on older versions of TexLive

# NOTES
# using plt.plot(visible=False ) will allow linestyle/marker to be included
# in legend but line will not plotted

linestyle_tuples = {
     'solid':                 '-',
     'dashdot':               '-.',
     
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     
     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}


def mathrm_label(label):
    label = r"$\mathrm{%s}$" % label.replace(" ", r"\;")
    label = label.replace("-", r"}$-$\mathrm{")
    return label


def mathrm_labels(labels):
    return [mathrm_label(label) for label in labels]

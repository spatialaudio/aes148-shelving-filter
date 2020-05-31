"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

Fig. 4c, slide 10/12

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from scipy.signal import sos2zpk
from util import low_shelving_2nd_cascade, shelving_filter_parameters, \
    shelving_slope_parameters, sosfreqs, db, set_rcparams, \
    set_outdir

set_rcparams()
outdir = set_outdir()

Biquad_per_octave = 2/3, 1, 2
labels = ['2/3', '1', '3']

w0 = 1
BWd = 9
Q = 1 / np.sqrt(2)
Gd = 10 * np.log10(0.5) * BWd
slope = shelving_slope_parameters(BWd=BWd, Gd=Gd)[0]

# Frequency-domain evaluation
wmin, wmax, num_w = 2**-9.5, 2**4.5, 1000
w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

# Filter design
shelving_filters = []
zpks = []
H = np.zeros((len(Biquad_per_octave), num_w), dtype='complex')
Gain = np.zeros(len(Biquad_per_octave))
for n, biquad_per_octave in enumerate(Biquad_per_octave):
    num_biquad, Gb, G = \
        shelving_filter_parameters(biquad_per_octave=biquad_per_octave,
                                   Gd=Gd, BWd=BWd)
    Gain[n] = G
    sos = low_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave)
    H[n] = sosfreqs(sos, worN=w)[1]
    shelving_filters.append(sos)
    zpks.append(sos2zpk(sos))

# desired response
wl, wh = w0 * 2**(-BWd), w0
Hmag = np.clip(np.log2(w/w0) * slope, G, 0)
Hmag = np.log2(w/w0) * slope

# Plots
Glim = -0.21, 0.21
philim = -3, 47
wlim = wmin, wmax
wticks = 2**(np.arange(np.ceil(np.log2(w)[0]/2)*2,
                       np.floor(np.log2(w[-1])/2)*2 + 2, 2))
kw = {'lw': 2, 'alpha': 1, 'basex': 2}
colors = cm.get_cmap('Blues')

# frequency response
fig, ax = plt.subplots(figsize=(10, 4), ncols=2, gridspec_kw={'wspace': 0.25})
for n, (biquad_per_octave, H_n) in enumerate(zip(Biquad_per_octave, H)):
    col = colors((n + 2) / (len(H) + 2))
    ax[0].semilogx(w, db(H_n) - Hmag, c=col, **kw,
                   label='{:0.0f}'.format(biquad_per_octave))

# Zeros and poles
kw_p = dict(c='k', marker='x', ls='none')
kw_z = dict(marker='o', mew=0.75, ls='none', mfc='none')
for n, (zpk) in enumerate(zpks):
    z, p, _ = zpk
    num_pole, num_zero = len(z), len(p)
    voffset = -n
    col = colors((n + 2) / (len(H) + 2))
    ax[1].plot(np.abs(p), voffset * np.ones(num_pole), **kw_p)
    ax[1].plot(np.abs(z), voffset * np.ones(num_zero), c=col, **kw_z)
ax[1].set_xscale('log', basex=2)

# decorations
ax[0].set_xlim(wmin, wmax)
ax[0].set_ylim(Glim)
ax[0].set_xticks(wticks)
ax[0].grid(True)
ax[0].yaxis.set_major_locator(MultipleLocator(0.1))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax[0].legend(labels, title='Biquad per octave', loc='upper right',
             facecolor='w', fontsize=10)
ax[0].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[0].set_ylabel('Level Error in dB')
ax[1].set_xlim(wmin, wmax)
ax[1].set_ylim(-2.5, 0.5)
ax[1].set_xticks(wticks)
ax[1].grid(True)
ax[1].yaxis.set_major_locator(MultipleLocator(1))
ax[1].set_yticks([0, -1, -2])
ax[1].set_yticklabels(labels)
ax[1].set_xlabel(r'$|s|$ / $\omega_\textrm{\footnotesize u}$')
ax[1].set_ylabel(r'Biquad per octave $N_O$')

# desired bandwidth
kw_bw = dict(color='lightgray', alpha=0.5)
Gmin, Gmax = ax[0].get_ylim()
ax[0].fill_between(x=(wl, wh), y1=Gmin, y2=Gmax, **kw_bw)
ax[1].fill_between(x=(wl, wh), y1=-4, y2=1, **kw_bw)
ax[0].text(np.sqrt(wl*wh) * 2**0.1, 0.19,
           'Bandwidth ${}$ oct'.format(BWd),
           va='top', ha='center', fontsize=10)

plt.savefig(outdir + 'ripple-and-zp-frequency.pdf', bbox_inches='tight')

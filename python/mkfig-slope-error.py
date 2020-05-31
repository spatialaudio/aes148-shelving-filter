"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

Fig. 4b,

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from util import low_shelving_2nd_cascade, shelving_filter_parameters, \
    shelving_slope_parameters, sosfreqs, db, set_rcparams, \
    set_outdir

set_rcparams()
outdir = set_outdir()

Biquad_per_octave = 1/3, 2/3, 1, 3
labels = ['1/3', '2/3', '1', '3']

w0 = 1
BWd = 3
wh = w0 * 2**(-0.5 / 6)
Q = 1 / np.sqrt(2)
Gd = -3 * 120 * np.log10(2)
slope = shelving_slope_parameters(BWd=BWd, Gd=Gd)[0]

# Frequency-domain evaluation
wmin, wmax, num_w = 2**-9.5, 2**4.5, 1000
w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

# Filter design
shelving_filters = []
H = np.zeros((len(Biquad_per_octave), num_w), dtype='complex')
Gain = np.zeros(len(Biquad_per_octave))
Gain_biquad = np.zeros(len(Biquad_per_octave))
for n, biquad_per_octave in enumerate(Biquad_per_octave):
    num_biquad, Gb, G = \
        shelving_filter_parameters(biquad_per_octave=biquad_per_octave,
                                   Gd=Gd, BWd=BWd)
    Gain[n] = G
    Gain_biquad[n] = Gb
    sos = low_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave)
    H[n] = sosfreqs(sos, worN=w)[1]
    shelving_filters.append(sos)

# Plots
wlim = wmin, wmax
wticks = 2**(np.arange(np.ceil(np.log2(w)[0]/2)*2,
                       np.floor(np.log2(w[-1])/2)*2 + 2, 2))
kw = {'lw': 2, 'alpha': 1, 'basex': 2}
colors = cm.get_cmap('Blues')

fig, ax = plt.subplots(figsize=(10, 4), ncols=2, gridspec_kw={'wspace': 0.25})

# frequency response
for n, H_n in enumerate(H):
    col = colors((n + 3) / (len(H) + 3))
    ax[0].semilogx(w, db(H_n), c=col, **kw)
    ax[1].semilogx(w, np.rad2deg(np.unwrap(np.angle(H_n))), c=col, **kw)

# desired response
wl, wh = w0 * 2**(-BWd), w0
Hmag = np.clip(np.log2(w/w0) * slope, G, 0)
Hphase = np.round(90 * slope / db(2), decimals=0)
ax[0].semilogx(w, Hmag, 'k:', **kw)
ax[1].semilogx((wl, w0), (Hphase, Hphase), 'k:', **kw)
ax[1].text(np.sqrt(wl * w0), Hphase - 10, '{:0.0f} deg'.format(Hphase),
           fontsize=10, ha='center', va='top')

# decorations
ax[0].set_xlim(wmin, wmax)
ax[0].set_xticks(wticks)
ax[0].grid(True)
ax[0].yaxis.set_major_locator(MultipleLocator(50))
ax[0].yaxis.set_minor_locator(MultipleLocator(10))
ax[0].legend(labels, title='Biquad \n per octave', loc='upper left',
             facecolor='w')
ax[0].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[0].set_ylabel('Level in dB')
ax[1].set_xlim(wmin, wmax)
ax[1].set_xticks(wticks)
ax[1].grid(True)
ax[1].yaxis.set_major_locator(MultipleLocator(180))
ax[1].yaxis.set_minor_locator(MultipleLocator(30))
ax[1].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[1].set_ylabel(r'Phase in degree')

# desired bandwidth
kw_bw = dict(color='lightgray', alpha=0.5)
Gmin, Gmax = ax[0].get_ylim()
phimin, phimax = ax[1].get_ylim()
ax[0].fill_between(x=(wl, wh), y1=Gmin, y2=Gmax, **kw_bw)
ax[1].fill_between(x=(wl, wh), y1=phimin, y2=phimax, **kw_bw)
ax[0].text(np.sqrt(wl*wh) * 2**0.1, 2, 'Bandwidth \n ${}$ oct'.format(BWd),
           va='top', ha='center', fontsize=10)
ax[0].set_ylim(Gmin, Gmax)
ax[1].set_ylim(phimin, phimax)

# desired slope
label = '{:+4.2f} dB/oct'.format(slope)
wc = w0 * 2**(-BWd/2)
w1 = wc * 2**(-0.2)
w2 = wc * 2**(+0.2)
Gc = Gd / 2
G1 = Gc + slope * -0.2
G2 = Gc + slope * 0.2
pc = ax[0].transData.transform_point((wc, Gc))
p1 = ax[0].transData.transform_point((w1, G1))
p2 = ax[0].transData.transform_point((w2, G2))
angle = np.rad2deg(np.arctan2(*(p2 - p1)[::-1]))
dx, dy = 30, 10
xoffset = (dx * np.cos(np.deg2rad(angle)),
           dx * np.sin(np.deg2rad(angle)))
yoffset = (dy * -np.sin(np.deg2rad(angle)),
           dy * np.cos(np.deg2rad(angle)))
inv = ax[0].transData.inverted()
(wt, Gt) = inv.transform(pc + xoffset + yoffset)
ax[0].annotate(label, (wt, Gt), ha='center', va='center',
               rotation=angle, fontsize=10)

plt.savefig(outdir + 'slope-error.pdf', bbox_inches='tight')

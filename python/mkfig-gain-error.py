"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

Fig. 4a, slide 9/12

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

Biquad_per_octave = 1, 2, 3, 6

w0 = 1
num, denom = 19, 6
BWd = num / denom
Q = 1 / np.sqrt(2)
Gd = 10 * np.log10(0.5) * BWd
slope = shelving_slope_parameters(BWd=BWd, Gd=Gd)[0]

# Frequency-domain evaluation
wmin, wmax, num_w = 2**-9.5, 2**4.5, 1000
w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

# Filter design
shelving_filters = []
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

# Plots
Glim = -13, 2
philim = -3, 47
wlim = wmin, wmax
wticks = 2**(np.arange(np.ceil(np.log2(w)[0]/2)*2,
                       np.floor(np.log2(w[-1])/2)*2 + 2, 2))
kw = {'lw': 2, 'alpha': 1, 'basex': 2}
labels = ['{:0.0f}'.format(biquad_per_octave)
          for biquad_per_octave in Biquad_per_octave]
colors = cm.get_cmap('Blues')

# frequency response
fig, ax = plt.subplots(figsize=(10, 4), ncols=2, gridspec_kw={'wspace': 0.25})
for n, (biquad_per_octave, H_n) in enumerate(zip(Biquad_per_octave, H)):
    col = colors((n + 3) / (len(H) + 3))
    ax[0].semilogx(w, db(H_n), c=col, **kw,
                   label='{:0.0f}'.format(biquad_per_octave))
    ax[1].semilogx(w, np.angle(H_n, deg=True), c=col, **kw)

# desired response
wl, wh = w0 * 2**(-BWd), w0
Hmag = np.clip(np.log2(w / w0) * slope, G, 0)
Hphase = 90 * slope / db(2)
ax[0].semilogx(w, Hmag, 'k:', **kw)
ax[1].semilogx((wl, w0), (Hphase, Hphase), 'k:', **kw)
ax[1].text(np.sqrt(wl * w0), Hphase - 1, '{:0.0f} deg'.format(Hphase),
           fontsize=10, ha='center', va='top')

# decorations
ax[0].set_xlim(wmin, wmax)
ax[0].set_ylim(Glim)
ax[0].set_xticks(wticks)
ax[0].grid(True)
ax[0].yaxis.set_major_locator(MultipleLocator(3))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(reversed(handles), reversed(labels),
             title='Biquad \n per octave', loc='upper left', facecolor='w')
ax[0].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[0].set_ylabel('Level in dB')
ax[1].set_xlim(wmin, wmax)
ax[1].set_ylim(philim)
ax[1].set_xticks(wticks)
ax[1].grid(True)
ax[1].yaxis.set_major_locator(MultipleLocator(10))
ax[1].yaxis.set_minor_locator(MultipleLocator(2))
ax[1].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[1].set_ylabel(r'Phase in degree')

# desired bandwidth
kw_bw = dict(color='lightgray', alpha=0.5)
Gmin, Gmax = ax[0].get_ylim()
ax[0].fill_between(x=(wl, wh), y1=Gmin, y2=Gmax, **kw_bw)
ax[1].fill_between(x=(wl, wh), y1=philim[0], y2=philim[-1], **kw_bw)
ax[0].text(np.sqrt(wl*wh) * 2**0.1, 1.7,
           'Bandwidth \n ${}/{}$ oct'.format(num, denom),
           va='top', ha='center', fontsize=10)

# desired slope
label = '{:+0.2f} dB/oct'.format(np.round(slope, decimals=2))
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
dx, dy = -5, 10
xoffset = (dx * np.cos(np.deg2rad(angle)),
           dx * np.sin(np.deg2rad(angle)))
yoffset = (dy * -np.sin(np.deg2rad(angle)),
           dy * np.cos(np.deg2rad(angle)))
inv = ax[0].transData.inverted()
(wt, Gt) = inv.transform(pc + xoffset + yoffset)
ax[0].annotate(label, (wt, Gt), ha='center', va='center',
               rotation=angle, fontsize=10)

# gain
w_G = 2**-3
for n, G in enumerate(Gain):
    col = colors((n + 3) / (len(H) + 3))
    label = '{:0.2f} dB'.format(np.round(G, decimals=2))
    if G == Gd:
        label += ' ($G$)'
    ax[0].text(w_G, G, label, color=col, va='center', ha='left', fontsize=9)

plt.savefig(outdir + 'gain-error.pdf', bbox_inches='tight')

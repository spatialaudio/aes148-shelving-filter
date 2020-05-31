"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

Fig. 3c, slide 8/12

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from util import low_shelving_2nd_cascade, shelving_filter_parameters,\
                 sosfreqs, db, shelving_slope_parameters, nearest_value,\
                 set_rcparams, set_outdir

set_rcparams()
outdir = set_outdir()

w0 = 1
Q = 1 / np.sqrt(2)

# Frequency-domain evaluation
wmin, wmax, num_w = 2**-9.5, 2**4.5, 1000
w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

# Different slopes
biquad_per_octave = 3
BWd = 6
Desired_gain = -8, -4, 3, 7
Slope = np.zeros(len(Desired_gain))

# Filter design
H = np.zeros((len(Desired_gain), num_w), dtype='complex')
for n, Gd in enumerate(Desired_gain):
    num_biquad, Gb, G = shelving_filter_parameters(
                biquad_per_octave=biquad_per_octave,
                BWd=BWd,
                Gd=Gd)
    Slope[n] = shelving_slope_parameters(BWd=BWd, Gd=Gd)[0]
    sos = low_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave)
    H[n] = sosfreqs(sos, worN=w)[1]

# Plots
Glim = -9.5, 9.5
philim = -21, 21
wlim = wmin, wmax
wticks = 2**(np.arange(np.ceil(np.log2(w)[0]/2)*2,
                       np.floor(np.log2(w[-1])/2)*2 + 2, 2))
kw_pos = dict(lw=1.5, c='C3', alpha=0.5, basex=2)
kw_neg = dict(lw=1.5, c='C0', alpha=1, basex=2)

fig, ax = plt.subplots(figsize=(10, 4), ncols=2, gridspec_kw={'wspace': 0.25})

# frequency response
w_mag, w_ph = 2**-9, 2**-3
for n, (Gd, H_n) in enumerate(zip(Desired_gain, H)):
    kw = kw_pos if Gd > 0 else kw_neg
#    col = 'C3' if Gd > 0 else 'C0'
    ax[0].semilogx(w, db(H_n), **kw)
    ax[1].semilogx(w, np.rad2deg(np.angle(H_n)), **kw)
    label = '{:+0.0f} dB'.format(Gd)
    phi = nearest_value(w_ph, w, np.angle(H_n, deg=True))
    ax[0].annotate(label, (w_mag, Gd), ha='left', va='bottom', fontsize=10)
    ax[1].annotate(label, (w_ph, phi), ha='center', va='bottom', fontsize=10)

# decorations
ax[0].set_xlim(wmin, wmax)
ax[0].set_ylim(Glim)
ax[0].set_xticks(wticks)
ax[0].grid(True)
ax[0].yaxis.set_major_locator(MultipleLocator(3))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
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
wl, wh = w0 * 2**(-BWd), w0
ax[0].text(np.sqrt(wl*wh), 8, 'Bandwidth {:0.0f} oct'.format(BWd),
           fontsize=10, va='bottom', ha='center')
Gmin, Gmax = ax[0].get_ylim()
phimin, phimax = ax[1].get_ylim()
ax[0].fill_between(x=(wl, wh), y1=Gmin-3, y2=Gmax+3, **kw_bw)
ax[1].fill_between(x=(wl, wh), y1=phimin-5, y2=phimax+5, **kw_bw)

# slope
for n, (Gd, sl) in enumerate(zip(Desired_gain, Slope)):
    label = '{:+0.2f} dB/oct'.format(np.round(sl, decimals=2))
    wc = w0 * 2**(-BWd/2)
    w1 = wc * 2**(-0.2)
    w2 = wc * 2**(+0.2)
    Gc = Gd / 2
    G1 = Gc + sl * -0.2
    G2 = Gc + sl * 0.2
    pc = ax[0].transData.transform_point((wc, Gc))
    p1 = ax[0].transData.transform_point((w1, G1))
    p2 = ax[0].transData.transform_point((w2, G2))
    angle = np.rad2deg(np.arctan2(*(p2-p1)[::-1]))
    dx, dy = -5, 7
    xoffset = (dx * np.cos(np.deg2rad(angle)),
               dx * np.sin(np.deg2rad(angle)))
    yoffset = (dy * -np.sin(np.deg2rad(angle)),
               dy * np.cos(np.deg2rad(angle)))
    inv = ax[0].transData.inverted()
    (wt, Gt) = inv.transform(pc + xoffset + yoffset)
    ax[0].annotate(label, (wt, Gt), ha='center', va='center',
                   rotation=angle, fontsize=10)

plt.savefig(outdir + 'low-shelving-filter-varying-gain.pdf',
            bbox_inches='tight')

"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

Fig. 3a, slide 6/12

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
Gd = 12
biquad_per_octave = 3
Slope = 10 * np.log10(2.**np.array([2, 1, 0.5, -0.5, -1, -2]))
Bandwidth = np.zeros(len(Slope))
Gain = np.zeros(len(Slope))

# Filter design
H = np.zeros((len(Slope), num_w), dtype='complex')
for n, slope in enumerate(Slope):
    num_biquad, Gb, G = shelving_filter_parameters(
                biquad_per_octave=biquad_per_octave, slope=slope,
                Gd=-Gd*np.sign(slope))
    Gain[n] = -Gd*np.sign(slope)
    Bandwidth[n] = shelving_slope_parameters(
            slope=slope, Gd=-Gd*np.sign(slope))[1]
    sos = low_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave)
    H[n] = sosfreqs(sos, worN=w)[1]

# Plots
wlim = wmin, wmax
Glim = -13, 14.8
philim = -54, 54
wticks = 2**(np.arange(np.ceil(np.log2(w)[0]/2)*2,
                       np.floor(np.log2(w[-1])/2)*2 + 2, 2))
kw_pos = dict(lw=1.5, c='C3', alpha=0.5, basex=2)
kw_neg = dict(lw=1.5, c='C0', alpha=1, basex=2)

fig, ax = plt.subplots(figsize=(10, 4), ncols=2, gridspec_kw={'wspace': 0.25})

# frequency response
for n, (sl, H_n) in enumerate(zip(Slope, H)):
    kw = kw_pos if Gain[n] > 0 else kw_neg

    ax[0].semilogx(w, db(H_n), **kw)
    ax[1].semilogx(w, np.angle(H_n, deg=True), **kw)

# decorations
ax[0].set_xlim(wmin, wmax)
ax[0].set_ylim(Glim)
ax[0].set_xticks(wticks)
ax[0].yaxis.set_major_locator(MultipleLocator(3))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
ax[0].grid(True)
ax[0].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[0].set_ylabel('Level in dB')
ax[1].set_xlim(wmin, wmax)
ax[1].set_ylim(philim)
ax[1].set_xticks(wticks)
ax[1].yaxis.set_major_locator(MultipleLocator(15))
ax[1].yaxis.set_minor_locator(MultipleLocator(5))
ax[1].grid(True)
ax[1].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[1].set_ylabel('Phase in degree')

# desired slope
for n, sl in enumerate(Slope):
    label = '{:+0.2f} dB/oct'.format(np.round(sl, decimals=2))
    wc = w0 * 2**(-Bandwidth[n]/2)
    w2 = wc * 2**(+0.2)
    w1 = wc * 2**(-0.2)
    Gc = Gain[n] / 2
    G2 = Gc + sl * 0.2
    G1 = Gc + sl * -0.2
    pc = ax[0].transData.transform_point((wc, Gc))
    p1 = ax[0].transData.transform_point((w1, G1))
    p2 = ax[0].transData.transform_point((w2, G2))
    angle = np.rad2deg(np.arctan2(*(p2-p1)[::-1]))
    dx, dy = -5, 6
    xoffset = (dx * np.cos(np.deg2rad(angle)),
               dx * np.sin(np.deg2rad(angle)))
    yoffset = (dy * -np.sin(np.deg2rad(angle)),
               dy * np.cos(np.deg2rad(angle)))
    inv = ax[0].transData.inverted()
    (wt, Gt) = inv.transform(pc + xoffset + yoffset)
    ax[0].annotate(label, (wt, Gt), ha='center', va='center',
                   rotation=angle, fontsize=10)

    w_phi = w[np.argmax(np.abs(np.angle(H[n])))]
    phi_label = nearest_value(w_phi, w, np.angle(H[n]))
    va = 'top' if sl < 0 else 'bottom'
    ax[1].annotate(label, (w_phi, np.rad2deg(phi_label)),
                   ha='right', va=va, fontsize=10)

# bandwidth
for n, (G, BW) in enumerate(zip(Gain[3:], Bandwidth[3:])):
    if G < 0:
        pass
    else:
        label = '{:0.0f} oct'.format(BW)
        wl, wh = 2**(-BW), w0
        G += (n+1) * 0.5
        ax[0].plot((wl, wh), (G, G), c='k', lw=1, marker='|')
        ax[0].annotate(s=label, xy=(wl * (wh/wl)**(1/8), G+0),
                       fontsize=8, va='bottom', ha='left')

plt.savefig(outdir + 'low-shelving-filter-varying-slope.pdf',
            bbox_inches='tight')

plt.savefig(outdir + 'low-shelving-filter-varying-slope.png',
            bbox_inches='tight')
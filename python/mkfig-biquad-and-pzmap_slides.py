"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

slide 4/12

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from scipy.signal import sos2zpk
from util import low_shelving_2nd_cascade, shelving_filter_parameters,\
                 shelving_slope_parameters, sosfreqs, db, set_rcparams,\
                 set_outdir

set_rcparams()
outdir = set_outdir()

w0 = 1
BWd = 3
Q = 1 / np.sqrt(2)
Gd = 10 * np.log10(0.5) * BWd
biquad_per_octave = 1
slope = shelving_slope_parameters(BWd=BWd, Gd=Gd)[0]
wl, wh = w0 * 2**(-BWd), w0

# Frequency-domain evaluation
wmin, wmax, num_w = 2**-9.5, 2**4.5, 1000
w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

# Filter design
num_biquad, Gb, G = shelving_filter_parameters(
        biquad_per_octave=biquad_per_octave, Gd=Gd, BWd=BWd)
Hmag = np.clip(np.log2(w/w0) * slope, G, 0)  # desired response
sos = low_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave)
H_biquads = np.stack([sosfreqs(sosi[np.newaxis, :], worN=w)[1]
                      for sosi in sos])
H = sosfreqs(sos, worN=w)[1]

# Plots
Glim = -9.5, 1
wlim = wmin, wmax
wticks = 2**(np.arange(np.ceil(np.log2(w)[0]/2)*2,
                       np.floor(np.log2(w[-1])/2)*2 + 2, 2))
kw = dict(lw=2, alpha=1, basex=2)
kw_dotted = dict(color='gray', linestyle=':', linewidth=1)
kw_artist = dict(edgecolor='gray', linestyle=':', linewidth=1)
colors = cm.get_cmap('Blues')

# frequency response
fig, ax = plt.subplots(figsize=(10, 4), ncols=2, gridspec_kw={'wspace': 0.1})
ax[0].semilogx(w, db(H_biquads.T), c='gray', **kw, zorder=2)
ax[0].semilogx(w, Hmag, 'k:', **kw)
ax[0].semilogx(w, db(H), c='C0', **kw, zorder=3)

ax[1].plot([-1, 0], [1, 0], 'C7:', lw=1)

# Pole zero plot
kw_z = dict(c='C0', marker='o', ms=9, ls='none', mew=1, mfc='none', alpha=1)
kw_p = dict(c='k', marker='x', ms=9, ls='none', mew=1)
kw_dot = dict(marker='.', ms=10)
ylim = ax[0].get_ylim()
for n, sosi in enumerate(sos):
    z, p, _ = sos2zpk(sosi[np.newaxis, :])
    z, p = z[0], p[0]
    wc = np.abs(np.sqrt(z * p))

    ax[0].plot((wc, wc), (-3, 0), **kw_dotted)
    ax[0].plot(wc, Gb/2, c='gray', zorder=2, **kw_dot)
    ax[1].plot(wc * -np.cos(np.pi/4), wc * np.sin(np.pi/4), c='gray', **kw_dot)
    ax[1].plot(np.real(p), np.imag(p), **kw_p)
    ax[1].plot(np.real(z), np.imag(z), **kw_z)
    circle = plt.Circle(xy=(0, 0), radius=wc, facecolor='none', **kw_artist)
    ax[1].add_artist(circle)
ax[0].plot(np.sqrt(wh * wl), G/2, 'C0', **kw_dot)

# desired slope
wc = 2**-2
w1 = wc * 2**(-0.2)
w2 = wc * 2**(+0.2)
Gc = Gd / 2
G1 = Gc + slope * -0.2
G2 = Gc + slope * +0.2
pc = ax[0].transData.transform_point((wc, Gc))
p1 = ax[0].transData.transform_point((w1, G1))
p2 = ax[0].transData.transform_point((w2, G2))
angle = np.rad2deg(np.arctan2(*(p2-p1)[::-1]))
dx, dy = -8, 0
xoffset = (dx * np.cos(np.deg2rad(angle)),
           dx * np.sin(np.deg2rad(angle)))
yoffset = (dy * -np.sin(np.deg2rad(angle)),
           dy * np.cos(np.deg2rad(angle)))
inv = ax[0].transData.inverted()
(wt, Gt) = inv.transform(pc + xoffset + yoffset)
ax[0].annotate('{:+0.2f} dB/oct'.format(np.round(slope, decimals=2)),
               (wt, Gt), ha='center', va='center', rotation=angle, fontsize=10)

# decorations
ax[0].set_xlim(wmin, wmax)
ax[0].set_ylim(Glim)
ax[0].set_xticks(wticks)
ax[0].grid(True)
ax[0].yaxis.set_major_locator(MultipleLocator(3))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
ax[0].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[0].set_ylabel('Level in dB')
ax[0].text(2**(-2.5), 0.5,
           r'upper cutoff frequency $\omega_\mathrm{u}$',
           fontsize=10, ha='center', va='center')

ax[1].axis([-0.75, 0., -0., 0.75])
ax[1].grid(True)
ax[1].set_xticks(-2.**-np.arange(3) * 10**(Gb/20))
ax[1].set_yticks(2.**-np.arange(3) * 10**(Gb/20))
ax[1].set_xticklabels(['$-2^{-1/2}$', '$-2^{-3/2}$', '$-2^{-5/2}$'])
ax[1].set_yticklabels(['$2^{-1/2}$', '$2^{-3/2}$', '$2^{-5/2}$'])
ax[1].yaxis.tick_right()
ax[1].set_aspect('equal')
ax[1].set_ylabel(r'$\Im (s\,/\,\omega_\textrm{\footnotesize u})$')
ax[1].set_xlabel(r'$\Re (s\,/\,\omega_\textrm{\footnotesize u})$')
ax[1].text(-2**(-2/2), 2**(-6/2),
           r'upper left $s$-plane', fontsize=10, ha='center', va='center')

plt.savefig(outdir + 'biquad-and-pzmap_slides.pdf', bbox_inches='tight')

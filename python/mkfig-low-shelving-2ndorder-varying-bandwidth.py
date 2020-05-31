"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

Fig. 3b, slide 7/12

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from util import low_shelving_2nd_cascade, shelving_filter_parameters,\
                 sosfreqs, db, nearest_value, set_rcparams, set_outdir

set_rcparams()
outdir = set_outdir()

w0 = 1
Q = 1 / np.sqrt(2)

# Frequency-domain evaluation
wmin, wmax, num_w = 2**-9.5, 2**4.5, 1000
w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

# Different slopes
biquad_per_octave = 3
slope = 10 * np.log10(2)
Desired_bandwidth = 1, 2, 3, 6, 8
Gain = np.zeros(len(Desired_bandwidth))

# Filter design
H = np.zeros((len(Desired_bandwidth), num_w), dtype='complex')
for n, BWd in enumerate(Desired_bandwidth):
    num_biquad, Gb, G = shelving_filter_parameters(
                biquad_per_octave=biquad_per_octave,
                slope=slope, BWd=BWd)
    Gain[n] = G
    sos = low_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave)
    H[n] = sosfreqs(sos, worN=w)[1]

# Plots
wlim = wmin, wmax
wticks = 2**(np.arange(np.ceil(np.log2(w)[0]/2)*2,
                       np.floor(np.log2(w[-1])/2)*2 + 2, 2))
kw = {'lw': 1.5, 'alpha': 0.75, 'basex': 2}

fig, ax = plt.subplots(figsize=(10, 4), ncols=2, gridspec_kw={'wspace': 0.25})

# frequency response
for n, (BWd, H_n) in enumerate(zip(Desired_bandwidth, H)):
    col = 'C3' if Gain[n] > 0 else 'C0'
    ax[0].semilogx(w, db(H_n), c=col, **kw)
    ax[1].semilogx(w, np.angle(H_n, deg=True), c=col, **kw)

# desired response
Hmag = np.clip(np.log2(w/w0) * slope, G, 0)
Hphase = 90 * slope / db(2)
ax[0].semilogx(w, Hmag, 'k:', **kw)
ax[1].semilogx((wmin, wmax), (Hphase, Hphase), 'k:', **kw)

# decorations
ax[0].set_xlim(wmin, wmax)
ax[0].set_xticks(wticks)
ax[0].yaxis.set_major_locator(MultipleLocator(3))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
ax[0].grid(True)
ax[0].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[0].set_ylabel('Level in dB')
ax[1].set_xlim(wmin, wmax)
ax[1].set_xticks(wticks)
ax[1].yaxis.set_major_locator(MultipleLocator(15))
ax[1].yaxis.set_minor_locator(MultipleLocator(5))
ax[1].grid(True)
ax[1].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize u}$')
ax[1].set_ylabel(r'Phase in degree')

# desired slope
wc = 2**-4
w1 = wc * 2**(-0.2)
w2 = wc * 2**(+0.2)
Gc = Gain[-1]/2
G1 = Gc + slope * -0.2
G2 = Gc + slope * +0.2
pc = ax[0].transData.transform_point((wc, Gc))
p1 = ax[0].transData.transform_point((w1, G1))
p2 = ax[0].transData.transform_point((w2, G2))
angle = np.rad2deg(np.arctan2(*(p2-p1)[::-1]))
dx, dy = -12, 7
xoffset = (dx * np.cos(np.deg2rad(angle)),
           dx * np.sin(np.deg2rad(angle)))
yoffset = (dy * -np.sin(np.deg2rad(angle)),
           dy * np.cos(np.deg2rad(angle)))
inv = ax[0].transData.inverted()
(wt, Gt) = inv.transform(pc + xoffset + yoffset)
ax[0].annotate('{:+0.2f} dB/oct'.format(np.round(slope, decimals=2)),
               (wt, Gt), ha='center', va='center', rotation=angle, fontsize=10)

# bandwidth
for n, (G, BWd) in enumerate(zip(Gain, Desired_bandwidth)):
    label = '{:0.0f} oct'.format(BWd)
    wl, wh = 2**(-BWd), w0
    ax[0].plot((wl, wh), (G, G), c='k', marker='|')
    ax[0].annotate(s=label, xy=(np.sqrt(wl*wh), Gain[n]+0.25),
                   fontsize=8, va='bottom')
    w_phi = w[np.argmax(np.abs(np.angle(H[n])))]
    phi_label = nearest_value(w_phi, w, np.angle(H[n]))
    va = 'top' if slope < 0 else 'bottom'
    ax[1].annotate(label, (w_phi, np.rad2deg(phi_label)),
                   ha='right', va=va, fontsize=10)

plt.savefig(outdir + 'low-shelving-filter-varying-bandwidth.pdf',
            bbox_inches='tight')

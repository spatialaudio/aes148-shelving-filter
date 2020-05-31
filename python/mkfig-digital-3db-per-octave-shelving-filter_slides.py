"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

slide 11/12

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bilinear_zpk, unit_impulse, sos2zpk, zpk2sos,\
                         sosfilt, sosfreqz
from matplotlib.ticker import MultipleLocator
from util import low_shelving_2nd_cascade, shelving_filter_parameters,\
                 db, set_rcparams, set_outdir, matchedz_zpk

set_rcparams()
outdir = set_outdir()


fc = 2000
w0 = 1
Q = 1 / np.sqrt(2)

# Frequency-domain evaluation
wmin, wmax, num_w = 2**-9.5, 2**4.5, 1000
w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

fmin, fmax, num_f = 10, 22000, 1000
f = np.logspace(np.log10(fmin), np.log10(fmax), num=num_f)

# Different slopes
biquad_per_octave = 1
slope = 10 * np.log10(2)
BWd = 6

# Time-domain evaluation
fs = 48000
ws = 2 * np.pi * fs
Lh = 1500
t = np.arange(Lh) / fs
xin = unit_impulse(Lh)
t = np.arange(Lh) / fs
s2z = matchedz_zpk
# s2z = bilinear_zpk

# Analog filter
H = np.zeros(num_w, dtype='complex')
num_biquad, Gb, G = shelving_filter_parameters(
            biquad_per_octave=biquad_per_octave,
            slope=slope, BWd=BWd)
sos_sdomain = low_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave)
zs, ps, ks = sos2zpk(sos_sdomain)

# Digital filter
zpk = s2z(zs * 2 * np.pi * fc, ps * 2 * np.pi * fc, ks, fs=fs)
sos_zdomain = zpk2sos(*zpk)
H = sosfreqz(sos_zdomain, worN=f, fs=fs)[1]
h = sosfilt(sos_zdomain, xin)

# Plots
flim = fmin, fmax
fticks = fc * 2.**np.arange(-8, 4, 2)
fticklabels = ['7.8', '31.3', '125', '500', '2k', '8k']
fticks = 1000 * 2.**np.arange(-6, 6, 2)
fticklabels = ['15.6', '62.5', '250', '1k', '4k', '16k']
kw = dict(c='C0', lw=2, alpha=1)

fig, ax = plt.subplots(figsize=(13, 3), ncols=3, gridspec_kw={'wspace': 0.25})

# frequency response
ax[0].semilogx(f, db(H), **kw)
ax[1].semilogx(f, np.angle(H, deg=True), **kw)

# desired response
fl, fh = fc * 2**(-BWd), fc
kw_des = dict(c='k', lw=2, ls=':')
Hmag = np.clip(np.log2(f/fc) * slope, G, 0)
Hphase = 90 * slope / db(2)
ax[0].semilogx(f, Hmag, **kw_des)
ax[1].semilogx((fl, fh), (Hphase, Hphase), **kw_des)
ax[1].text(250, Hphase-1, '45 degree', va='top', ha='center', fontsize=10)

# desired bandwidth
kw_bw = dict(color='lightgray', alpha=0.5)
fl, fh = fc * 2**(-BWd), fc
Gmin, Gmax = ax[0].get_ylim()
phimin, phimax = ax[1].get_ylim()
ax[0].fill_between(x=(fl, fh), y1=Gmin-3, y2=Gmax+3, **kw_bw)
ax[1].fill_between(x=(fl, fh), y1=phimin-5, y2=phimax+5, **kw_bw)

# Pole zero plot
kw_z = dict(c='C0', marker='o', ms=9, ls='none', mew=1, mfc='none', alpha=1)
kw_p = dict(c='k', marker='x', ms=9, ls='none', mew=1)
kw_artist = dict(edgecolor='gray', linestyle='-', linewidth=1)
z, p, _ = zpk
ax[2].plot(np.real(p), np.imag(p), **kw_p)
ax[2].plot(np.real(z), np.imag(z), **kw_z)
circle = plt.Circle(xy=(0, 0), radius=1, facecolor='none', **kw_artist)
ax[2].add_artist(circle)

# decorations
ax[0].set_xlim(flim)
ax[0].set_ylim(Gmin, Gmax)
ax[0].set_xticks(fticks)
ax[0].set_xticklabels(fticklabels)
ax[0].minorticks_off()
ax[0].yaxis.set_major_locator(MultipleLocator(3))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
ax[0].grid(True)
ax[0].set_xlabel('Frequency in Hz')
ax[0].set_ylabel('Level in dB')
ax[1].set_xlim(flim)
ax[1].set_ylim(phimin, phimax)
ax[1].set_xticks(fticks)
ax[1].set_xticklabels(fticklabels)
ax[1].minorticks_off()
ax[1].yaxis.set_major_locator(MultipleLocator(15))
ax[1].yaxis.set_minor_locator(MultipleLocator(5))
ax[1].grid(True)
ax[1].set_xlabel('Frequency in Hz')
ax[1].set_ylabel(r'Phase in degree')
ax[2].axis([0.76, 1.04, -0.14, 0.14])
ax[2].grid(True)
ax[2].set_aspect('equal')
ax[2].set_ylabel(r'$\Im (z)$')
ax[2].set_xlabel(r'$\Re (z)$')
ax[2].xaxis.set_major_locator(MultipleLocator(0.1))
ax[2].yaxis.set_major_locator(MultipleLocator(0.1))
ax[2].text(0.85, -0.05, r'$z$-plane', fontsize=10, ha='center', va='center')

# desired slope
f0 = fc * 2**(-BWd/2)
f1 = fc * 2**(-0.2)
f2 = fc * 2**(+0.2)
G0 = G / 2
G1 = G0 + slope * -0.2
G2 = G0 + slope * +0.2
p0 = ax[0].transData.transform_point((f0, G0))
p1 = ax[0].transData.transform_point((f1, G1))
p2 = ax[0].transData.transform_point((f2, G2))
angle = np.rad2deg(np.arctan2(*(p2-p1)[::-1]))
dx, dy = 0, 7
xoffset = (dx * np.cos(np.deg2rad(angle)),
           dx * np.sin(np.deg2rad(angle)))
yoffset = (dy * -np.sin(np.deg2rad(angle)),
           dy * np.cos(np.deg2rad(angle)))
inv = ax[0].transData.inverted()
(ft, Gt) = inv.transform(p0 + xoffset + yoffset)
ax[0].annotate('{:+0.2f} dB/oct'.format(np.round(slope, decimals=2)),
               (ft, Gt), ha='center', va='center', rotation=angle, fontsize=10)

plt.savefig(outdir + 'digital-3db-per-octave-shelving-filter_slides.pdf',
            bbox_inches='tight')

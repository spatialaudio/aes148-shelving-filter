"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

slide 3/12

design by Holters, M. and Zölzer, U.
"Parametric Recursive Higher-Order Shelving Filters,"
In: Proc of 120th AES Convention, Paris, 2006, Paper 6722

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from scipy.signal import freqs_zpk
from util import db, set_outdir, set_rcparams


def higher_order_shelving_holters(Gd, M, wc=1, normalize=True):
    g = 10**(Gd / 20)
    alpha = np.stack([np.pi * (0.5 - (2*m+1)/2/M) for m in range(M)])
    p = -np.exp(1j * alpha)
    z = g**(1 / M) * p
    k = 1
    if normalize:
        z *= g**(-0.5 / M)
        p *= g**(-0.5 / M)
    return z * wc, p * wc, k


set_rcparams()
outdir = set_outdir()

wc = 2**0
Gd = -11
max_order = 6
orders = np.arange(1, max_order+1)
wmin, wmax, num_w = 2**-9.5, 2**4.5, 1000
w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

H = np.zeros((max_order, len(w)), dtype='complex')
for i, M in enumerate(orders):
    zpk = higher_order_shelving_holters(Gd, M, wc=wc, normalize=True)
    H[i] = freqs_zpk(*zpk, worN=w)[1]
z, p, _ = zpk

wlim = w[0], w[-1]
wticks = 2**(np.arange(np.ceil(np.log2(w)[0]/4)*4,
                       np.floor(np.log2(w[-1])/4)*4 + 4, 4))
kw = dict(linewidth=2, alpha=1, basex=2)
kw_z = dict(c='C0', marker='o', ms=9, ls='none', mew=1, mfc='none', alpha=1)
kw_p = dict(c='k', marker='x', ms=9, ls='none', mew=1)
kw_artist = dict(edgecolor='gray', linestyle=':', linewidth=1)
colors = [cm.get_cmap('Oranges')(x)[:3]
          for x in np.linspace(0.33, 1, num=max_order, endpoint=False)]

fig, ax = plt.subplots(figsize=(13, 3), ncols=3, gridspec_kw={'wspace': 0.3})

for Hi, ci in zip(H, colors):
    ax[0].semilogx(w / wc, db(Hi), c=ci, **kw)
    ax[1].semilogx(w / wc, np.angle(Hi, deg=True), c=ci, **kw)
ax[0].set_xlim(wlim)
ax[0].set_xticks(wticks)
ax[0].grid(True)
ax[0].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize c}$')
ax[0].set_ylabel('Level in dB')
ax[0].yaxis.set_major_locator(MultipleLocator(3))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
ax[0].legend(orders, title='Filter order', facecolor='w')

ax[1].set_xlim(wlim)
ax[1].set_xticks(wticks)
ax[1].grid(True)
ax[1].set_xlabel(r'$\omega$ / $\omega_\textrm{\footnotesize c}$')
ax[1].set_ylabel('Phase in degree')
ax[1].yaxis.set_major_locator(MultipleLocator(30))
ax[1].yaxis.set_minor_locator(MultipleLocator(5))
ax[1].text(2**(-2.5), 20,
           r'cutoff frequency $\omega_\mathrm{c}$',
           fontsize=10, ha='center', va='center')

ax[2].plot(np.real(p), np.imag(p), **kw_p)
ax[2].plot(np.real(z), np.imag(z), **kw_z)
ax[2].axis([-1.2, 1.2, -1.2, 1.2])
circle = plt.Circle(xy=(0, 0), radius=1, facecolor='none', **kw_artist)
ax[2].add_artist(circle)
ax[2].grid(True)
ax[2].set_aspect('equal')
ax[2].yaxis.set_major_locator(MultipleLocator(1))
ax[2].yaxis.set_minor_locator(MultipleLocator(0.5))
ax[2].set_xlabel(r'$\Re (s\,/\,\omega_\textrm{\footnotesize c})$')
ax[2].set_ylabel(r'$\Im (s\,/\,\omega_\textrm{\footnotesize c})$')
ax[2].text(0, 0.05, 'Filter order: 6', fontsize=10, ha='center', va='center')
ax[2].text(0.5, -0.5, r'$s$-plane', fontsize=10, ha='center', va='center')

plt.savefig(outdir + 'holters2006-higher-order-shelving_slides.pdf',
            bbox_inches='tight')

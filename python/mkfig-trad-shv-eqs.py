"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

slide 1/12

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import freqs
from util import low_shelving_2nd_coeff
from util import high_shelving_2nd_coeff
from util import set_outdir, set_rcparams

basex = 10  # semilogx base, 2 or 10

set_rcparams()
outdir = set_outdir()

kw = dict(lw=3, alpha=0.75)

# frequency axis
fmin, fmax, num_f = 9, 22000, 5000
f = np.logspace(np.log10(fmin), np.log10(fmax), num=num_f)
fticks = 1000 * 2.**np.arange(-6, 6, 2)
fticklabels = ['15.6', '62.5', '250', '1k', '4k', '16k']
dBticks = np.arange(-18, 18+6, 6)

fc = 500
wc = 2*np.pi*fc
w = 2*np.pi*f

fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
ax[0].plot((fc, fc), (-18, +18), c='gold')
ax[1].plot((fc, fc), (-18, +18), c='gold')

for G in range(-12, 0, 3):
    Q = 1/np.sqrt(2)
    b, a = low_shelving_2nd_coeff(omega=wc, G=+G, Q=Q)
    w, H = freqs(b, a, w)
    ax[0].semilogx(f, 20*np.log10(np.abs(H)), **kw, c='C0')
    b, a = high_shelving_2nd_coeff(omega=wc, G=-G, Q=Q)
    w, H = freqs(b, a, w)
    ax[0].semilogx(f, 20*np.log10(np.abs(H)), **kw, c='C1')

for G in range(-12, 0, 12):
    Q = 1/np.sqrt(2)
    b, a = high_shelving_2nd_coeff(omega=wc, G=-G, Q=Q)
    w, H = freqs(b, a, w)
    ax[1].semilogx(f, 20*np.log10(np.abs(H)), '--', **kw, c='C1',
                   label='G=%+2.f dB, Q=%4.3f' % (-G, Q))
    Q = 0.5
    b, a = high_shelving_2nd_coeff(omega=wc, G=-G, Q=Q)
    w, H = freqs(b, a, w)
    ax[1].semilogx(f, 20*np.log10(np.abs(H)), **kw, c='C1',
                   label='G=%+2.f dB, Q=%3.1f' % (-G, Q))
    Q = 1/np.sqrt(2)
    b, a = low_shelving_2nd_coeff(omega=wc, G=+G, Q=Q)
    w, H = freqs(b, a, w)
    ax[1].semilogx(f, 20*np.log10(np.abs(H)), '--', **kw, c='C0',
                   label='G=%+2.f dB, Q=%4.3f' % (+G, Q))
    Q = 2
    b, a = low_shelving_2nd_coeff(omega=wc, G=+G, Q=Q)
    w, H = freqs(b, a, w)
    ax[1].semilogx(f, 20*np.log10(np.abs(H)), **kw, c='C0',
                   label='G=%+2.f dB, Q=%2.f' % (+G, Q))


ax[0].grid(True)
ax[0].set_xlim(f[0], f[-1])
ax[0].set_xticks(fticks)
ax[0].set_xticklabels(fticklabels)
ax[0].minorticks_off()
ax[0].set_ylim(-18, 18)
ax[0].set_yticks(dBticks)
ax[0].yaxis.set_major_locator(MultipleLocator(3))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
ax[0].set_xlabel('Frequency in Hz')
ax[0].set_ylabel('Level in dB')
ax[0].set_title('Varying Level, Fixed Q=0.7071')

ax[0].text(20, +15, 'High Shelving Filter Orange')
ax[0].text(20, -15, 'Low Shelving Filter Blue')

ax[1].grid(True)
ax[1].set_xlim(f[0], f[-1])
ax[1].set_xticks(fticks)
ax[1].set_xticklabels(fticklabels)
ax[1].minorticks_off()
ax[1].set_ylim(-18, 18)
ax[1].set_yticks(dBticks)
ax[1].yaxis.set_major_locator(MultipleLocator(3))
ax[1].yaxis.set_minor_locator(MultipleLocator(1))
ax[1].set_xlabel('Frequency in Hz')
ax[1].set_ylabel('Level in dB')
ax[1].set_title('Varying Q')
ax[1].legend()

plt.savefig(outdir + 'trad-shv-eqs.pdf', bbox_inches='tight')

"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

Fig. 6

design by Liski, J., Rämö, J., and Välimäki, V.
"Graphic Equalizer Design with Symmetric Biquad Filters,"
In: Proc. of IEEE Workshop Appl. Sig. Process. Audio Acoust. (WASPAA)
pp. 55–59, New Paltz, 2019.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import sosfreqz
from util import db, set_rcparams, set_outdir, optimized_peq_seg

set_rcparams()
outdir = set_outdir()

fs = 48000

# frequency axis
fmin, fmax, num_f = 9, 22000, 5000
f = np.logspace(np.log10(fmin), np.log10(fmax), num=num_f)
fticks = 1000 * 2.**np.arange(-6, 6, 2)
fticklabels = ['15.6', '62.5', '250', '1k', '4k', '16k']

# command and control frequencies
f_command = 16000 * 2.**np.arange(-9, 1)
num_band = len(f_command)
f_control = np.zeros(2 * num_band - 1)
f_control[::2] = f_command
f_control[1::2] = np.sqrt(f_command[:-1] * f_command[1:])
w_command = 2 * np.pi * f_command / fs
w_control = 2 * np.pi * f_control / fs
bandwidth = 1.5 * w_command
bandwidth[6] *= 0.997
bandwidth[7] *= 0.985
bandwidth[8] *= 0.929
bandwidth[9] *= 0.433

gain_factor = 0.29
gain_proto = np.array([13.8, 14.5, 14.5, 14.6, 14.5,
                       14.5, 14.6, 14.6, 14.5, 13.6])

# 3dB per octave
slope = db(np.sqrt(2))
BWd = 6
fh = 2000
fl = fh * 2**(-BWd)
f_center = fh * 2**(-BWd/2)
w_center = 2 * np.pi * f_center / fs
gmin, gmax = -9, 9
gain_command = np.clip(np.log2(w_command / w_center) * slope, gmin, gmax)
H_desired = np.clip(np.log2(f / f_center) * slope, gmin, gmax)
b_opt, a_opt = optimized_peq_seg(
        gain_command, gain_proto, gain_factor, w_command, w_control, bandwidth)
sos = np.vstack([b_opt, a_opt]).T
H = sosfreqz(sos, worN=f, fs=fs)[1]


# Plots
kw = dict(c='peru', lw=3, alpha=0.75)
kw_desired = dict(c='k', lw=2, ls=':', alpha=1)
kw_command = dict(marker='o', ms=8, mew=0, mfc='r', alpha=0.5, ls='')

fig, ax = plt.subplots(figsize=(4, 3.5))

ax.semilogx(f, db(H), **kw)
ax.semilogx(f, H_desired, **kw_desired)
ax.semilogx(f_command, gain_command, **kw_command)
ax.grid(True)
ax.set_xlim(fmin, fmax)
ax.set_xticks(fticks)
ax.set_xticklabels(fticklabels)
ax.minorticks_off()
ax.yaxis.set_major_locator(MultipleLocator(3))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.set_xlabel('Frequency in Hz')
ax.set_ylabel('Level in dB')

# desired bandwidth
kw_bw = dict(color='lightgray', alpha=0.5)
Gmin, Gmax = ax.get_ylim()
ax.fill_between(x=(fl, fh), y1=Gmin, y2=Gmax, **kw_bw)
ax.set_ylim(Gmin, Gmax)

# desired slope
label = '{:+0.2f} dB/oct'.format(np.round(slope, decimals=2))
f1 = f_center * 2**(-0.2)
f2 = f_center * 2**(+0.2)
Gc = 0
G1 = Gc + slope * -0.2
G2 = Gc + slope * 0.2
pc = ax.transData.transform_point((f_center, Gc))
p1 = ax.transData.transform_point((f1, G1))
p2 = ax.transData.transform_point((f2, G2))
angle = np.rad2deg(np.arctan2(*(p2-p1)[::-1]))
dx, dy = 0, 10
xoffset = (dx * np.cos(np.deg2rad(angle)),
           dx * np.sin(np.deg2rad(angle)))
yoffset = (dy * -np.sin(np.deg2rad(angle)),
           dy * np.cos(np.deg2rad(angle)))
inv = ax.transData.inverted()
(wt, Gt) = inv.transform(pc + xoffset + yoffset)
ax.annotate(label, (wt, Gt), ha='center', va='center',
            rotation=angle, fontsize=10)

plt.savefig(outdir + 'liski2019-geq-3db-per-oct.pdf', bbox_inches='tight')

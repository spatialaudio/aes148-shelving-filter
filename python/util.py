"""Shelving Filter Cascade with Adjustable Transition Slope and Bandwidth
Frank Schultz, Nara Hahn, Sascha Spors
In: Proc. of 148th AES Convention, Virtual Vienna, May 2020, Paper 10339
http://www.aes.org/e-lib/browse.cfm?elib=20756

"""
import numpy as np
from scipy.signal import tf2sos, freqs
from matplotlib import rcParams


def halfpadloss_shelving_filter_num_den_coeff(G):
    """Half-pad-loss polynomial coefficients for 1st/2nd order shelving filter.

    - see type III in
    long-url: https://github.com/spatialaudio/digital-signal-processing-lecture/blob/master/filter_desig/audiofilter.ipynb  # noqa

    - see Sec. 3.2 in https://doi.org/10.3390/app6050129

    """
    sign = np.sign(G)  # amplify/boost (1) or attenuate/cut (-1)
    g = 10**(np.abs(G) / 20)  # linear gain
    n1, n2 = g**(sign / 4), g**(sign / 2)  # numerator coeff
    d1, d2 = 1 / n1, 1 / n2  # denominator coeff
    return n1, n2, d1, d2


def normalized_low_shelving_1st_coeff(G=-10*np.log10(2)):
    """See low_shelving_1st_coeff() for omega=1."""
    n1, n2, d1, d2 = halfpadloss_shelving_filter_num_den_coeff(G)
    b, a = np.array([0, 1, n2]), np.array([0, 1, d2])
    return b, a


def low_shelving_1st_coeff(omega=1, G=-10*np.log10(2)):
    """Half-pad-loss/mid-level low shelving filter 1st order.

    Parameters
    ----------
    omega : angular frequency in rad/s at half-pad-loss/mid-level
    G : level in dB (G/2 at omega)
    Returns
    -------
                                          b[0] s^2 + b[1] s^1 + b[2] s^0
    b,a : coefficients for Laplace H(s) = ------------------------------
                                          a[0] s^2 + a[1] s^1 + a[2] s^0
    with s = j omega, note: b[0]=a[0]=0 here for 1st order filter

    see halfpadloss_shelving_filter_num_den_coeff() for references

    """
    b, a = normalized_low_shelving_1st_coeff(G=G)
    scale = omega**np.arange(-2., 1.)  # powers in the Laplace domain
    return b * scale, a * scale


def normalized_high_shelving_1st_coeff(G=-10*np.log10(2)):
    """See high_shelving_1st_coeff() for omega=1."""
    n1, n2, d1, d2 = halfpadloss_shelving_filter_num_den_coeff(G)
    b, a = np.array([0, n2, 1]), np.array([0, d2, 1])
    return b, a


def high_shelving_1st_coeff(omega=1, G=-10*np.log10(2)):
    """Half-pad-loss/mid-level high shelving filter 1st order.

    Parameters
    ----------
    omega : angular frequency in rad/s at half-pad-loss/mid-level
    G : level in dB (G/2 at omega)
    Returns
    -------
                                          b[0] s^2 + b[1] s^1 + b[2] s^0
    b,a : coefficients for Laplace H(s) = ------------------------------
                                          a[0] s^2 + a[1] s^1 + a[2] s^0
    with s = j omega, note: b[0]=a[0]=0 here for 1st order filter

    see halfpadloss_shelving_filter_num_den_coeff() for references

    """
    b, a = normalized_high_shelving_1st_coeff(G=G)
    scale = omega**np.arange(-2., 1.)  # powers in the Laplace domain
    return b * scale, a * scale


def normalized_low_shelving_2nd_coeff(G=-10*np.log10(2), Q=1/np.sqrt(2)):
    """See low_shelving_2nd_coeff() for omega=1."""
    n1, n2, d1, d2 = halfpadloss_shelving_filter_num_den_coeff(G)
    b, a = np.array([1, n1 / Q, n2]), np.array([1, d1 / Q, d2])
    return b, a


def low_shelving_2nd_coeff(omega=1, G=-10*np.log10(2), Q=1/np.sqrt(2)):
    """Half-pad-loss/mid-level low shelving filter 2nd order.

    Parameters
    ----------
    omega : angular frequency in rad/s at half-pad-loss/mid-level
    G : level in dB (G/2 at omega)
    Q : pole/zero quality, Q>0.5
    Returns
    -------
                                          b[0] s^2 + b[1] s^1 + b[2] s^0
    b,a : coefficients for Laplace H(s) = ------------------------------
                                          a[0] s^2 + a[1] s^1 + a[2] s^0
    with s = j omega

    see halfpadloss_shelving_filter_num_den_coeff() for references

    """
    b, a = normalized_low_shelving_2nd_coeff(G=G, Q=Q)
    scale = omega**np.arange(-2., 1.)  # powers in the Laplace domain
    return b * scale, a * scale


def normalized_high_shelving_2nd_coeff(G=-10*np.log10(2), Q=1/np.sqrt(2)):
    """See high_shelving_2nd_coeff() for omega=1."""
    n1, n2, d1, d2 = halfpadloss_shelving_filter_num_den_coeff(G)
    b, a = np.array([n2, n1 / Q, 1]), np.array([d2, d1 / Q, 1])
    return b, a


def high_shelving_2nd_coeff(omega=1, G=-10*np.log10(2), Q=1/np.sqrt(2)):
    """Half-pad-loss/mid-level high shelving filter 2nd order.

    Parameters
    ----------
    omega : angular frequency in rad/s at half-pad-loss/mid-level
    G : level in dB (G/2 at omega)
    Q : pole/zero quality, Q>0.5
    Returns
    -------
                                          b[0] s^2 + b[1] s^1 + b[2] s^0
    b,a : coefficients for Laplace H(s) = ------------------------------
                                          a[0] s^2 + a[1] s^1 + a[2] s^0
    with s = j omega

    see halfpadloss_shelving_filter_num_den_coeff() for references

    """
    b, a = normalized_high_shelving_2nd_coeff(G=G, Q=Q)
    scale = omega**np.arange(-2., 1.)  # powers in the Laplace domain
    return b * scale, a * scale


def db(x, *, power=False):
    """Convert *x* to decibel.

    Parameters
    ----------
    x : array_like
        Input data.  Values of 0 lead to negative infinity.
    power : bool, optional
        If ``power=False`` (the default), *x* is squared before
        conversion.

    """
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


def db2lin(x):
    return 10**(x / 20)


def shelving_slope_parameters(slope=None, BWd=None, Gd=None):
    """Compute the third parameter from the given two.

    Parameters
    ----------
    slope : float, optional
        Desired shelving slope in decibel per octave.
    BW : float, optional
        Desired bandwidth of the slope in octave.
    G : float, optional
        Desired gain of the stop band in decibel.

    """
    if slope == 0:
        raise ValueError("`slope` should be nonzero.")
    if slope and BWd is not None:
        Gd = -BWd * slope
    elif BWd and Gd is not None:
        slope = -Gd / BWd
    elif Gd and slope is not None:
        if Gd * slope > 1:
            raise ValueError("`Gd` and `slope` cannot have the same sign.")
        else:
            BWd = np.abs(Gd / slope)
    else:
        print('At lest two parameters need to be specified.')
    return slope, BWd, Gd


def shelving_filter_parameters(biquad_per_octave, **kwargs):
    """Parameters for shelving filter design.

    Parameters
    ----------
    biquad_per_octave : float
        Number of biquad filters per octave.

    Returns
    -------
    num_biquad : int
        Number of biquad filters.
    Gb : float
        Gain of each biquad filter in decibel.
    G : float
        Gain of overall (concatenated) filters in decibel. This might differ
        from what is returned by `shelving_parameters`.

    """
    slope, BWd, Gd = shelving_slope_parameters(**kwargs)
    num_biquad = int(np.ceil(BWd * biquad_per_octave))
    Gb = -slope / biquad_per_octave
    G = Gb * num_biquad
    return num_biquad, Gb, G


def check_shelving_filter_validity(biquad_per_octave, **kwargs):
    """Level, slope, bandwidth validity for shelving filter cascade.

    Parameters
    ----------
    biquad_per_octave : float
        Number of biquad filters per octave.

    see shelving_slope_parameters(), shelving_filter_parameters()

    Returns
    -------
    flag = [Boolean, Boolean, Boolean]

    if all True then intended parameter triplet holds, if not all True
    deviations from desired response occur

    """
    flag = [True, True, True]
    slope, BWd, Gd = shelving_slope_parameters(**kwargs)
    num_biquad, Gb, G = shelving_filter_parameters(biquad_per_octave, **kwargs)

    # BWd < 1 octave generally fails
    if BWd <= 1:
        flag[0] = False

    # BWd * biquad_per_octave needs to be integer
    flag[1] = float(BWd * biquad_per_octave).is_integer()

    # biquad_per_octave must be large enough
    # for slope < 12.04 dB at least one biquad per ocatve is required
    tmp = slope / (20*np.log10(4))
    if tmp > 1.:
        if biquad_per_octave < tmp:
            flag[2] = False
    else:
        if biquad_per_octave < 1:
            flag[2] = False
    return flag


def low_shelving_1st_cascade(w0, Gb, num_biquad, biquad_per_octave):
    """Low shelving filter design using cascaded biquad filters.

    - see low_shelving_2nd_cascade()
    - under construction for code improvement

    """
    sos = np.zeros((num_biquad, 6))
    for m in range(num_biquad):
        wm = w0 * 2**(-(m + 0.5) / biquad_per_octave)
        b, a = low_shelving_1st_coeff(omega=wm, G=Gb)
        sos[m] = tf2sos(b, a)
    return sos


def high_shelving_1st_cascade(w0, Gb, num_biquad, biquad_per_octave):
    """High shelving filter design using cascaded biquad filters.

    - see low_shelving_2nd_cascade()
    - under construction for code improvement

    """
    sos = np.zeros((num_biquad, 6))
    for m in range(num_biquad):
        wm = w0 * 2**(-(m + 0.5) / biquad_per_octave)
        b, a = high_shelving_1st_coeff(omega=wm, G=Gb)
        sos[m] = tf2sos(b, a)
    return sos


def low_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave,
                             Q=1/np.sqrt(2)):
    """Low shelving filter design using cascaded biquad filters.

    Parameters
    ----------
    w0 : float
        Cut-off frequency in radian per second.
    Gb : float
        Gain of each biquad filter in decibel.
    num_biquad : int
        Number of biquad filters.
    Q : float, optional
        Quality factor of each biquad filter.

    Returns
    -------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.

    """
    sos = np.zeros((num_biquad, 6))
    for m in range(num_biquad):
        wm = w0 * 2**(-(m + 0.5) / biquad_per_octave)
        b, a = low_shelving_2nd_coeff(omega=wm, G=Gb, Q=Q)
        sos[m] = tf2sos(b, a)
    return sos


def high_shelving_2nd_cascade(w0, Gb, num_biquad, biquad_per_octave,
                              Q=1/np.sqrt(2)):
    """High shelving filter design using cascaded biquad filters.

    - see low_shelving_2nd_cascade()
    - under construction for code improvement

    """
    sos = np.zeros((num_biquad, 6))
    for m in range(num_biquad):
        wm = w0 * 2**(-(m + 0.5) / biquad_per_octave)
        b, a = high_shelving_2nd_coeff(omega=wm, G=Gb, Q=Q)
        sos[m] = tf2sos(b, a)
    return sos


def sosfreqs(sos, worN=200, plot=None):
    """Compute the frequency response of an analog filter in SOS format.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations).  If a single
        integer, then compute at that many frequencies.  Otherwise, compute the
        response at the angular frequencies (e.g. rad/s) given in `worN`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqs`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    """
    h = 1.
    for row in sos:
        w, rowh = freqs(row[:3], row[3:], worN=worN, plot=plot)
        h *= rowh
    return w, h


def matchedz_zpk(s_zeros, s_poles, s_gain, fs):
    """Matched-z transform of poles and zeros.

    Parameters
    ----------
    s_zeros : array_like
        Zeros in the Laplace domain.
    s_poles : array_like
        Poles in the Laplace domain.
    s_gain : float
        System gain in the Laplace domain.
    fs : int
        Sampling frequency in Hertz.

    Returns
    -------
    z_zeros : numpy.ndarray
        Zeros in the z-domain.
    z_poles : numpy.ndarray
        Poles in the z-domain.
    z_gain : float
        System gain in the z-domain.

    See Also
    --------
    :func:`scipy.signal.bilinear_zpk`

    """
    z_zeros = np.exp(s_zeros / fs)
    z_poles = np.exp(s_poles / fs)
    omega = 1j * np.pi * fs
    s_gain *= np.prod((omega - s_zeros) / (omega - s_poles)
                      * (-1 - z_poles) / (-1 - z_zeros))
    return z_zeros, z_poles, np.abs(s_gain)


def nearest_value(x0, x, f):
    """Plot helping."""
    return f[np.abs(x - x0).argmin()]


def set_rcparams():
    """Plot helping."""
    rcParams['axes.linewidth'] = 0.5
    rcParams['axes.edgecolor'] = 'black'
    rcParams['axes.facecolor'] = 'None'
    rcParams['axes.labelcolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 13
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    rcParams['text.latex.preamble'] = r'\usepackage{gensymb}'
    rcParams['legend.title_fontsize'] = 10


def set_outdir():
    """Plot helping."""
    return '../graphics/'


def interaction_matrix_sge(G_proto, gain_factor, w_command, w_control,
                           bandwidth):
    """
    Parameters
    ----------
    G_proto: array_like
        Prototype gain in decibel.
    gain_factor: float
        Gain factor.
    w_command: array_like
        Normalized command frequencies.
    w_control: array_like
        Normalized control frequencies.
    bandwidth: array_like
        Bandwidth.
    """
    num_command = len(w_command)
    num_control = len(w_control)
    leak = np.zeros((num_command, num_control))
    G_bandwidth = gain_factor * G_proto
    g_proto = db2lin(G_proto)
    g_bandwidth = db2lin(G_bandwidth)

    z1 = np.exp(-1j * w_control)
    z2 = z1**2

    poly = np.zeros((num_command, 3))
    poly[6] = 0.000321, 0.00474, 0.00544
    poly[7] = 0.00108, 0.0221, 0.0169
    poly[8] = 0.00184, 0.125, 0.0212
    poly[9] = -0.00751, 0.730, -0.0672

    for m, (Gp, gp, p, gb, wc, bw) in enumerate(
            zip(G_proto, g_proto, poly, g_bandwidth, w_command, bandwidth)):
        G_nyquist = np.sign(Gp) * np.polyval(p, np.abs(Gp))
        gn = db2lin(G_nyquist)
        gp2 = gp**2
        gb2 = gb**2
        gn2 = gn**2

        F = np.abs(gp2 - gb2)

        G00 = np.abs(gp2 - 1)
        F00 = np.abs(gb2 - 1)

        G01 = np.abs(gp2 - gn)
        G11 = np.abs(gp2 - gn2)
        F01 = np.abs(gb2 - gn)
        F11 = np.abs(gb2 - gn2)

        W2 = np.sqrt(G11 / G00) * np.tan(wc / 2)**2
        DW = (1 + np.sqrt(F00 / F11) * W2) * np.tan(bw / 2)
        C = F11 * DW**2 - 2 * W2 * (F01 - np.sqrt(F00 * F11))
        D = 2 * W2 * (G01 - np.sqrt(G00 * G11))
        A = np.sqrt((C + D) / F)
        B = np.sqrt((gp2 * C + gb2 * D) / F)
        num = np.array([gn+W2+B, -2*(gn-W2), (gn-B+W2)]) / (1+W2+A)
        den = np.array([1, -2*(1-W2)/(1+W2+A), (1+W2-A)/(1+W2+A)])
        H = (num[0] + num[1]*z1 + num[2]*z2)\
            / (den[0] + den[1]*z1 + den[2]*z2)
        G = db(H) / Gp
        leak[m] = np.abs(G)
    return leak


def peq_seg(g_ref, g_nyquist, g, g_bandwidth, w_command, bandwidth):
    """
    Parameters
    ----------
    g_ref: float
        Reference linear gain.
    g_nyquist: float
        Nyquist linear gain.
    g_bandwidth: float
        (Optimized) linear gain.
    w_command: float
        Normalized command frequencies.
    bandwidth: float
        Bandwidth.
    """
    g2 = g**2
    gb2 = g_bandwidth**2
    gr2 = g_ref**2
    gn2 = g_nyquist**2
    grn = g_ref * g_nyquist

    F = np.abs(g2 - gb2)
    G00 = np.abs(g2 - gr2)
    F00 = np.abs(gb2 - gr2)

    G01 = np.abs(g2 - grn)
    G11 = np.abs(g2 - gn2)
    F01 = np.abs(gb2 - grn)
    F11 = np.abs(gb2 - gn2)

    W2 = np.sqrt(G11 / G00) * np.tan(w_command / 2)**2
    DW = (1 + np.sqrt(F00 / F11) * W2) * np.tan(bandwidth / 2)

    C = F11 * DW**2 - 2 * W2 * (F01 - np.sqrt(F00 * F11))
    D = 2 * W2 * (G01 - np.sqrt(G00 * G11))

    A = np.sqrt((C + D) / F)
    B = np.sqrt((g**2 * C + g_bandwidth**2 * D) / F)

    b = np.array([(g_nyquist + g_ref * W2 + B),
                  -2*(g_nyquist - g_ref * W2),
                  (g_nyquist - B + g_ref * W2)]) / (1 + W2 + A)
    a = np.array([1, -2*(1 - W2) / (1 + W2 + A), (1 + W2 - A) / (1 + W2 + A)])
    return b, a


def optimized_peq_seg(gain_command, gain_proto, gain_factor, w_command,
                      w_control, bandwidth):
    """
    Parameters
    ----------
    gain_command: array_like
        Command gain in decibel.
    gain_proto: array_like
        Prototype gain in decibel.
    gain_factor: float
        Gain factor.
    w_command: array_like
        Normalized command frequencies.
    w_control: array_like
        Normalized control frequencies.
    bandwidth: array_like
        Bandwidths.

    Returns
    -------
    b_opt: array_like (N, 3)
        Moving average coefficients.
    a_opt: array_like (N, 3)
        Autoregressive (recursive) coefficients.
    """
    num_command = len(gain_command)

    # symmetric GEG design
    gain_control = np.zeros(2 * num_command - 1)
    gain_control[::2] = gain_command
    gain_control[1::2] = 0.5 * (gain_command[:-1] + gain_command[1:])

    # interaction matrix "B"
    B = interaction_matrix_sge(gain_proto, gain_factor,
                               w_command, w_control, bandwidth)

    gain2 = np.zeros((2 * num_command - 1, 1))
    gain2[::2, 0] = gain_command
    gain2[1::2, 0] = 0.5 * (gain_command[:-1] + gain_command[1:])

    # band weights
    weights = np.ones(2 * num_command - 1)
    weights[1::2] *= 0.5
    W = np.diag(weights)

    gain_opt =\
        np.matmul(np.linalg.inv(np.linalg.multi_dot([B, W, np.transpose(B)])),
                  np.linalg.multi_dot([B, W, gain2]))
    gain_opt_bandwidth = gain_factor * gain_opt

    gain_opt = np.squeeze(gain_opt)
    gain_opt_bandwidth = np.squeeze(gain_opt_bandwidth)

    g_opt = db2lin(gain_opt)
    g_opt_bandwidth = db2lin(gain_opt_bandwidth)

    poly = np.zeros((num_command, 3))
    poly[6] = 0.000321, 0.00474, 0.00544
    poly[7] = 0.00108, 0.0221, 0.0169
    poly[8] = 0.00184, 0.125, 0.0212
    poly[9] = -0.00751, 0.730, -0.0672

    b_opt = np.zeros((3, num_command))
    a_opt = np.zeros((3, num_command))
    for m, (Go, go, gob, wc, bw, p) in enumerate(
            zip(gain_opt, g_opt, g_opt_bandwidth, w_command, bandwidth, poly)):
        gain_nyquist = np.sign(Go) * np.polyval(p, np.abs(Go))
        b, a = peq_seg(1, db2lin(gain_nyquist), go, gob, wc, bw)
        b_opt[:, m] = b
        a_opt[:, m] = a
    return b_opt, a_opt


def fracorder_lowshelving_eastty(w1, w2, G1, G2, rB=None):
    """
    Parameters
    ----------
    w1: float
        Lower corner frequency.
    w2: float
        Upper corner frequency.
    G1: float
        Target level at lower corner frequency in dB.
    G2: float
        Target level at upper corner frequency in dB.
    rB: float
        Gain per octave.

    Returns
    -------
    z: array_like
        Complex zeros in the Laplace domain.
    p: array_like
        Complex poles in the Laplace domain.
    k: float
        Gain.
    """
    Gd = G1 - G2
    n_eff = effective_order(w1, w2, Gd, rB)
    n_int, n_frac = np.divmod(n_eff, 1)
    n_int = int(n_int)
    z = np.array([])
    p = np.array([])

    # Second-order sections (complex conjugate pole/zero pairs)
    if n_int > 0:
        alpha = complex_zp_angles(n_int, n_frac)
        alpha = np.concatenate((alpha, -alpha))
        z = w1 * np.exp(1j * alpha)
        p = w2 * np.exp(1j * alpha)

    # First-order section (real pole/zero)
    if n_eff % 2 != 0:
        s_lower, s_upper = real_zp(n_int, n_frac, w1, w2)
        if n_int % 2 == 0:
            z_real = s_lower
            p_real = s_upper
        elif n_int % 2 == 1:
            z_real = s_upper
            p_real = s_lower
        z = np.append(z, z_real)
        p = np.append(p, p_real)
    return z, p, 1


def effective_order(w1, w2, Gd, rB=None):
    """Effective order of shelving filter.

    Parameters
    ----------
    w1: float
        Lower corner frequency.
    w2: float
        Upper corner frequency.
    Gd: float
        Target level difference in dB.
    rB: float
        Gain per octave.
    """
    if rB is None:
        rB = db(2) * np.sign(Gd)  # Butterworth
    return Gd / rB / np.log2(w2/w1)


def complex_zp_angles(n_int, n_frac):
    """Polar angles of the complex conjugate zeros/poles.
    These correspond to the second-order section filters.

    Parameters
    ----------
    n_int: int
        Interger order.
    n_frac: float
        Fractional order [0, 1).
    """
    # linear interpolation of angles
    num_zp_pair = int(n_int+1) // 2
    return np.pi/2 * np.stack([
            (1-n_frac) * (1 + (2*m+1)/n_int)
            + n_frac * (1 + (2*m+1)/(n_int+1))
            for m in range(num_zp_pair)])


def real_zp(n_int, n_frac, w_lower, w_upper):
    """Real-valued zero and pole.
    These correspond to the first-order section filters.

    Parameters
    ----------
    n_int: int
        Integer order
    n_frac: float
        Fractional order [0, 1).
    w_lower: float
        Lower corner frequency.
    w_upper: float
        Upper corner frequency.

    Returns
    -------
    s_lower: float
        Smaller real-valued zero or pole.
    s_upper: float
        Larger real-valued zero or pole.
    """
    w_mean = np.sqrt(w_lower * w_upper)
    ratio = (w_upper / w_lower)

    # logarithmic interpolation of zero/pole radius
    if n_int % 2 == 0:  # even
        s_lower = -w_mean * ratio**(-n_frac/2)
        s_upper = -w_mean * ratio**(n_frac/2)
    elif n_int % 2 == 1:  # odd
        s_lower = -w_lower * ratio**(n_frac/2)
        s_upper = -w_upper * ratio**(-n_frac/2)
    return s_lower, s_upper

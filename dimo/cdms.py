# import modules
import os
import sys
import math
import glob
import numpy as np
import time
import pandas as pd
from astropy import constants, units
import re
from io import StringIO

#from __future__ import annotations
from fractions import Fraction
from dataclasses import dataclass
from typing import Optional, List, Dict, Sequence, Tuple


# constants (in cgs)

Ggrav  = constants.G.cgs.value        # Gravitational constant
ms     = constants.M_sun.cgs.value    # Solar mass (g)
ls     = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
rs     = constants.R_sun.cgs.value    # Solar radius (cm)
au     = units.au.to('cm')            # 1 au (cm)
pc     = units.pc.to('cm')            # 1 pc (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
hp     = constants.h.cgs.value        # Planck constant
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mp     = constants.m_p.cgs.value      # Proton mass (g)

# path to here
path_to_here = os.path.dirname(__file__)
#print(path_to_here)
path_to_library = path_to_here + '/'
path_to_qt = path_to_library + 'moldata/parfunc_cdms.txt'
path_to_wt = path_to_library + 'moldata/molweights.csv'


QN_SCHEMES = {
    1: ["J"],
    2: ["N", "J", "F"],
    3: ["J", "Ka", "Kc"],
    4: ["J", "K"],
    5: ["J", "Ka", "Kc", "F"],
}

COLUMN_MAPPING = {
    "freq": 0,
    "e_freq": 1,
    "Smu2": 2,
    "DOG": 3,
    "Elow": 4,
    "gup": 5,
    "tag": 6,
    "qnfmt": 7,
}

mw = pd.read_csv(path_to_wt, sep=' ')

EN_DASH = "–"

class CDMS():
    '''
    A Python Class to read and sort molecular data from
    The Cologne Database for Molecular Spectroscopy (CDMS; https://cdms.astro.uni-koeln.de/).
    This Python class reads .cat file exported from CDMS.
    It assumes that the output is not in intensity but in S mu^2.
    Data columns are supposed to be

    frequency | e_freq | S mu^2 | degree of freedom | E_low | g_up | tag | coding of quantum number| Up | Low | Molecule |
    (MHz)     |  (MHz) |        |                   | cm^-1 | 
    '''

    def __init__(self, mol):
        self.line = mol
        #self.column_names = [
        #'freq', 'e_freq', 'Smu2' ,'DOG', 
        #'Elow', 'gup', 'tag', 'qnfmt', 
        #'qn_up', 'qn_low', 'Mol']
        self.read_cdms_moldata()
        self.weight = mw.loc[mw['tag'] == int(self.tag), 'weight'].iloc[0]
        #self.partition_grid(self.moltag)


    def partition_grid(self,):
        moltag = self.tag.zfill(6) # width of tag column
        df = read_partition_function(path_to_qt)
        Tgrid = np.array(
            [float(i.split('(')[-1].split(')')[0]) for i in df.columns if 'lg' in i]
            )
        Q_cols = [i for i in df.columns if 'lg' in i]
        _df = df[df['tag'] == moltag]
        #if mol is not None:
        #    _df = _df[_df['molecule'].str.contains(mol)]
        log10Qgrid = np.squeeze(_df[Q_cols].values)

        # sort
        sortindx = Tgrid.argsort()
        Tgrid = Tgrid[sortindx]
        log10Qgrid = log10Qgrid[sortindx]
        # remove nan
        where_nan = np.isnan(log10Qgrid)
        Tgrid = Tgrid[~where_nan]
        log10Qgrid = log10Qgrid[~where_nan]

        self.Tgrid = Tgrid
        self.Qgrid = 10**log10Qgrid
        #return Tgrid, 10**log10Qgrid


    def read_cdms_moldata(self):
        '''
        Read a molecular data file from LAMDA (Leiden Atomic and Molecular Database)
        '''
        # find path
        line = self.line.lower()
        if '+' in line: line = line.replace('+','p')
        infile = glob.glob(path_to_library+'moldata/'+line+'.cat')
        if len(infile) == 0:
            print('ERROR\tread_cdms_moldata: Cannot find CDMS file.')
            return
        else:
            data = read_cat(infile[0])

        nrows, ncols = data.shape
        self.tag = str(data['tag'][0])
        self.ntrans = nrows
        self.nlevels = nrows + 1
        itrans = data.index.values
        self.itrans = itrans

        freq = data['freq_MHz'].values * 1.e-3 # GHz
        Smusq = data['lgint'].values
        gup = data['gup'].values
        self.freq = freq
        self.Smusq = Smusq
        self.gup = gup


        # Get quantum numbers
        if data['QNFMT'].nunique() == 1:
            qnfmt = decode_qn_token(str(data['QNFMT'][0]))
        else:
            print('QNFMT is not unique among all transitions.')
            print('Cannot handle in the current CDMS version.')
            return 0
        fmt = parse_qnfmt(qnfmt)
        Q, H, NQN = fmt['Q'], fmt['H'], fmt['NQN']
        labels = QN_SCHEMES[Q][:NQN]
        #qnindx0 = COLUMN_MAPPING['qnfmt'] + 1
        #qnindx1 = COLUMN_MAPPING['qnfmt'] + 1 + NQN
        #i_qn_up = list(range(qnindx0, qnindx0 + NQN))
        #i_qn_low = list(range(qnindx1, qnindx1 + NQN))
        qn_up = {labels[i]: data[labels[i] + '_up'].values for i in range(NQN)}
        qn_low = {labels[i]: data[labels[i] + '_low'].values for i in range(NQN)}

        self.qnfmt = data['QNFMT'][0]
        self.qlables = labels
        self.qn_up = qn_up
        self.qn_low = qn_low
        self.nlevels = nrows

        # energy on each excitation level
        Elow = data['elo_cm-1'].values.copy()
        Eup = Elow + freq * 1.e9 / clight # in unit of cm^-1
        dE = freq * 1.e9 / clight # in unit of cm^-1
        self.Elow  = Elow * clight * hp / kb # from cm^-1 to K
        self.Eup = Eup * clight * hp / kb # from cm^-1 to K
        self.dE = dE * clight * hp / kb # from cm^-1 to K

        # Especially about J transitions
        if 'J' in labels:
            self.J = np.append(qn_low['J'], qn_up['J'][-1])
            self.Jlow = qn_low['J']
            self.Jup = qn_up['J']
            # omit cuz it leads an error. The transitions are not in order of J transitions but of frequency
            #self.gJ = 2 * self.J + 1
            #self.EJ = np.append(self.Elow, self.Eup[-1])

        # number of transition
        Acoeff = Smusq_to_Acoeff(self.Smusq, self.freq * 1.e9, self.gup)
        self.Acoeff = Acoeff


        # transitions
        #trans = [ str(J[ int(Jup[i] - 1)]) + '-' \
        #+ str( J[ int(Jlow[i] - 1)]) for i in range(len(itrans))]
        trans = [
        format_transition(
            data.qn_up[i], data.qn_lo[i], data.QNFMT[i]
            ) for i in range(nrows)]
        self.trans = trans
        #self.catrows = data


    def params_trans(self, iline, freq=False):
        '''
        Return parameters of a transition specified by iline.
        iline is the index of the transition in the catalog 
         if 'freq = False', or the frequency of the transition in GHz
         if 'freq = True'.

        Params
        ------
         iline (int): Index or frequency of the transition.
         freq (bool): If true, iline will be treated as frequency in GHz.
        '''
        if freq:
            iline = np.argmin((self.freq - iline)**2.) + 1 # get index

        # line Ju --> Jl
        Ju = self.Jup[iline-1]
        Jl = self.Jlow[iline-1]

        trans = self.trans[iline-1]
        freq = self.freq[iline-1] * 1e9 # Hz
        Aul     = self.Acoeff[iline-1]
        gu      = 2 * Ju + 1
        gl      = 2 * Jl + 1
        Eu     = self.Eup[iline-1]
        El     = self.Elow[iline-1]
        return trans, freq, Aul, gu, gl, Eu, El



def Smusq_to_Acoeff(Smusq, nu, gu, gJ = 1, gK = 1, gI = 1, ):
    '''
    Equation (11) of Mangum & Shirley (2015). Eq. (33) of the same paper
    provides the total degeneracy gu = gJ x gK x gI.

    Parameters
    ----------
    Smusq (float or array): S mu^2, where S is the line strength 
     and mu is the dipole moment. S dimensionless, mu in D^2 (D is Debye and 1D = 10^-18 esu cm).
    nu (float or array): Frequency (Hz)
    gJ (float): Rotational degeneracy gJ = 2J + 1.
    gK (float): K degeneracy associated with the internal quantum number K.
    gI (float): Nuclear spin degeneracy.

    Return
    ------
     Acoeff: Einstein A coefficient.
    '''
    if gu is None:
        gu = gJ * gK * gI
    return 64. * np.pi**4 * nu**3. / 3. / hp / clight**3. * Smusq * 1.e-36 / gu



# --- quantum-number token decoding (2-character token -> int or None) ---

def decode_qn_token(tok: str) -> Optional[int]:
    tok = tok.strip()
    if tok == "" or tok == "**":
        return None

    # plain integer 0..99 (may come with leading spaces)
    if tok.isdigit():
        return int(tok)

    # encoded >=100 : A0..Z9
    if len(tok) == 2 and tok[0].isupper() and tok[1].isdigit():
        return (ord(tok[0]) - ord("A") + 10) * 10 + int(tok[1])

    # encoded negatives -10..-259 : a0..z9
    if len(tok) == 2 and tok[0].islower() and tok[1].isdigit():
        return -((ord(tok[0]) - ord("a") + 10) * 10 + int(tok[1]))

    # sometimes you may see "+" or "-" forms in other tools; handle cautiously
    try:
        return int(tok)
    except ValueError:
        raise ValueError(f"Unrecognized QN token: {tok!r}")


def split_qn_field(qn12chars: str, nqn: int) -> List[Optional[int]]:
    """
    qn12chars: 12-character quantum number field (6 tokens x 2 chars)
    nqn: number of quantum numbers per state (from QNFMT)
    """
    tokens = [qn12chars[i:i+2] for i in range(0, 12, 2)]
    vals = [decode_qn_token(t) for t in tokens]
    return vals[:nqn]  # keep only those actually used for this species

# --- parse QNFMT into (Q, H, NQN) ---

def parse_qnfmt(qnfmt: int) -> Dict[str, int]:
    # QNFMT = Q*100 + H*10 + NQN
    Q = qnfmt // 100
    H = (qnfmt % 100) // 10
    NQN = qnfmt % 10
    return {"Q": Q, "H": H, "NQN": NQN}
    #return Q, H, NQN

# --- fixed-width line slicing (Pickett/JPL style) ---

@dataclass
class CatRow:
    freq_mhz: float
    err_mhz: float
    lgint: float
    dr: int
    elo_cm1: float
    gup: int
    tag: int
    qnfmt: int
    qn_up: List[Optional[int]]
    qn_lo: List[Optional[int]]
    mol: str

def parse_cat_line(line: str) -> CatRow:
    # ensure line is long enough
    if len(line) < 93:
        line = line.rstrip("\n")
        line = line + " " * (93 - len(line))

    freq = float(line[0:13])
    err  = float(line[13:24])
    lgint = float(line[24:35])
    dr = int(line[35:37])
    elo = float(line[37:47])
    gup = int(line[47:50])
    tag = int(line[50:57])
    qnfmt = int(line[57:61])

    qn_up_raw = line[61:73]   # 12 chars (6 tokens)
    qn_lo_raw = line[73:85]   # 12 chars

    fmt = parse_qnfmt(qnfmt)
    nqn = fmt["NQN"]
    #q, h, nqn = parse_qnfmt(qnfmt)

    qn_up = split_qn_field(qn_up_raw, nqn)
    qn_lo = split_qn_field(qn_lo_raw, nqn)

    mol = line[85:] if len(line) >= 93 else ''

    return CatRow(freq, err, lgint, dr, elo, gup, tag, qnfmt, qn_up, qn_lo, mol)


def read_cat(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            if not line.strip():
                continue
            r = parse_cat_line(line)
            fmt = parse_qnfmt(r.qnfmt)
            Q, H, NQN = fmt['Q'], fmt['H'], fmt['NQN']
            labels = QN_SCHEMES[Q][:NQN]
            row = {
                "freq_MHz": r.freq_mhz,
                "err_MHz": r.err_mhz,
                "lgint": r.lgint,
                "dr": r.dr,
                "elo_cm-1": r.elo_cm1,
                "gup": r.gup,
                "tag": r.tag,
                "QNFMT": r.qnfmt,
                "Q": fmt["Q"],
                "H": fmt["H"],
                "NQN": fmt["NQN"],
                "qn_up": r.qn_up,
                "qn_lo": r.qn_lo}
            for i in range(NQN):
                row.update({labels[i] + '_up': r.qn_up[i]})
                row.update({labels[i] + '_low': r.qn_lo[i]})
            rows.append(row)
    return pd.DataFrame(rows)


def Q_from_cdms_table(T, Tgrid, log10Qgrid):
    """
    T: float or array (K)
    Tgrid: array of temperatures from CDMS (K)
    log10Qgrid: array of log10(Q) at Tgrid
    Returns Q(T)
    """
    T = np.asarray(T, dtype=float)
    x = np.log(Tgrid)
    y = log10Qgrid
    yT = np.interp(np.log(T), x, y)  # linear in log(T), log10(Q)
    return 10.0**yT


def read_partition_function(f):
    tag_width = 6
    # 1) Read raw text
    with open(f, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 2) Keep only data lines (skip headers / separators)
    data_lines = []
    for line in lines:
        # data lines start with a TAG number
        if re.match(r"^\s*\d+\s+\S+", line):
            # insert 00 if needed to keep the tag width correct
            line = line[:tag_width].strip().zfill(tag_width) + line[tag_width:]
            data_lines.append(line.rstrip())

    data = "\n".join(data_lines)

    # header
    head_line = lines[0][:-2] # omit \n
    column_names = [ i for i in head_line.split(' ') if len(i) >= 1]

    # 3) Read as fixed-width table
    df = pd.read_fwf(StringIO(data), header = None, names = column_names, dtype = str)

    # 4) Convert lg(Q) columns to numeric
    for col in df.columns:
        if str(col).startswith("lg(Q"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def format_transition(qn_up: list,
                      qn_lo: list,
                      qnfmt: int,
                      *,
                      include_labels: bool = False) -> str:
    """
    Auto-format transition strings from qn_up/qn_lo + QNFMT.

    Supports common schemes:
      Q=1: linear rotor -> "Ju–Jl"
      Q=2: linear + (fine/hyperfine) -> labeled chain
      Q=3: asymmetric rotor -> "J_Ka,Kc–J'_Ka',Kc'"
      Q=4: symmetric rotor -> "J_K–J'_K'"
      Q=5: asymmetric rotor + F (common case) -> include F after semicolon if present

    Notes:
      - For many molecules, the exact meaning of QN slots can be more complex (torsion, parity, etc.).
      - This covers the dominant CDMS use-cases in astronomy (CO, H2CO, CH3CN, C17O, etc.).
    """
    fmt = parse_qnfmt(qnfmt)
    Q, H, NQN = fmt['Q'], fmt['H'], fmt['NQN']

    up_f = _apply_half_integer_flags(qn_up, H)
    lo_f = _apply_half_integer_flags(qn_lo, H)

    # Helper to join "_" and "," in the common spectroscopy style
    def rotor_triplet(u, l) -> str:
        # expects (J, Ka, Kc)
        Ju, Kau, Kcu = u
        Jl, Kal, Kcl = l
        return f"{_fmt_qn(Ju)}_{_fmt_qn(Kau)},{_fmt_qn(Kcu)}{EN_DASH}{_fmt_qn(Jl)}_{_fmt_qn(Kal)},{_fmt_qn(Kcl)}"

    def rotor_doublet(u, l) -> str:
        # expects (J, K)
        Ju, Ku = u
        Jl, Kl = l
        return f"{_fmt_qn(Ju)}_{_fmt_qn(Ku)}{EN_DASH}{_fmt_qn(Jl)}_{_fmt_qn(Kl)}"

    # --- Format by scheme ---
    if Q == 1:
        # linear rotor: usually NQN=1 with J
        if len(up_f) >= 1 and len(lo_f) >= 1:
            return f"{_fmt_qn(up_f[0])}{EN_DASH}{_fmt_qn(lo_f[0])}"
        return f"?{EN_DASH}?"

    if Q == 3:
        # asymmetric rotor: (J, Ka, Kc) plus possibly extra tokens (parity/state)
        if len(up_f) >= 3 and len(lo_f) >= 3:
            base = rotor_triplet(up_f[:3], lo_f[:3])
            if not include_labels and NQN == 3:
                return base
            # If extra QNs exist, append them in a compact way:
            extras_u = ",".join(_fmt_qn(x) for x in up_f[3:])
            extras_l = ",".join(_fmt_qn(x) for x in lo_f[3:])
            if extras_u or extras_l:
                return f"{base} ; extra={extras_u}{EN_DASH}{extras_l}"
            return base
        return f"?_{'?','?'}{EN_DASH}?_{'?','?'}"

    if Q == 4:
        # symmetric rotor: (J, K)
        if len(up_f) >= 2 and len(lo_f) >= 2:
            base = rotor_doublet(up_f[:2], lo_f[:2])
            if not include_labels and NQN == 2:
                return base
            extras_u = ",".join(_fmt_qn(x) for x in up_f[2:])
            extras_l = ",".join(_fmt_qn(x) for x in lo_f[2:])
            if extras_u or extras_l:
                return f"{base} ; extra={extras_u}{EN_DASH}{extras_l}"
            return base
        return f"?_?{EN_DASH}?_?"

    if Q == 2:
        # linear + fine/hyperfine often (N, J, F) or (N, J) depending on NQN.
        # We print labeled chain to be unambiguous.
        labels = ["N", "J", "F", "F1", "F2"]  # best-effort labels
        parts = []
        for i in range(min(len(up_f), len(lo_f))):
            lab = labels[i] if i < len(labels) else f"q{i+1}"
            parts.append(f"{lab}={_fmt_qn(up_f[i])}{EN_DASH}{_fmt_qn(lo_f[i])}")
        return ", ".join(parts)

    if Q == 5:
        # common extension: (J, Ka, Kc, F) best-effort
        if len(up_f) >= 3 and len(lo_f) >= 3:
            base = rotor_triplet(up_f[:3], lo_f[:3])
            if len(up_f) >= 4 and len(lo_f) >= 4:
                base += f" (F={_fmt_qn(up_f[3])}{EN_DASH}{_fmt_qn(lo_f[3])})"
            return base
        return f"?{EN_DASH}?"

    # Fallback: just print raw sequence
    up_s = ",".join(_fmt_qn(x) for x in up_f)
    lo_s = ",".join(_fmt_qn(x) for x in lo_f)
    return f"[{up_s}]{EN_DASH}[{lo_s}]"



def _apply_half_integer_flags(qns: Sequence[Optional[int]], H: int) -> list[Optional[Fraction]]:
    """
    In CDMS/JPL .cat encoding, half-integers are stored "rounded up".
    If a quantum number is half-integer, the stored integer n represents n - 1/2.
    Convention: H=0 none, H=1 last is half-int, H=2 last two, H=3 last three.
    """
    out: list[Optional[Fraction]] = [None if v is None else Fraction(v, 1) for v in qns]
    if H <= 0:
        return out
    k = min(H, len(out))
    for i in range(1, k + 1):
        idx = -i
        if out[idx] is not None:
            out[idx] = out[idx] - Fraction(1, 2)
    return out

def _fmt_qn(v: Optional[Fraction]) -> str:
    if v is None:
        return "?"
    if v.denominator == 1:
        return str(v.numerator)
    # e.g. 3/2
    return f"{v.numerator}/{v.denominator}"
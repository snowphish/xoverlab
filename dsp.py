import os, math, re
import numpy as np
from constants import FREQ_MIN_DEFAULT, FREQ_MAX_DEFAULT

# --------- helpers ---------
def wrap_phase_deg(ph_deg): 
    return ((ph_deg + 180.0) % 360.0) - 180.0

def unwrap_deg(ph_deg): 
    return np.rad2deg(np.unwrap(np.deg2rad(ph_deg)))

# ---------- REW parsing ----------
_num_re = re.compile(r'[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?')
def _extract_numbers(line): 
    return [float(m.replace(',', '.')) for m in _num_re.findall(line)]

def parse_rew_txt(path, fmin_for_accept=1, fmax_for_accept=200000):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    data=[]
    for s in lines:
        s=s.strip()
        if not s or s.startswith(("#",";","//","*")): continue
        nums=_extract_numbers(s)
        if len(nums)<2: continue
        f=nums[0]
        if not (fmin_for_accept<=f<=fmax_for_accept): continue
        m=nums[1]; p=nums[2] if len(nums)>=3 else 0.0
        data.append((f,m,p))
    if not data: 
        raise RuntimeError("No numeric data parsed. Try REW: Export → Measurement as TXT.")
    arr=np.array(data,dtype=float); arr=arr[np.argsort(arr[:,0])]
    _,idx=np.unique(arr[:,0],return_index=True); arr=arr[np.sort(idx)]
    arr=arr[(arr[:,0]>0)]
    return arr[:,0],arr[:,1],arr[:,2]

def resample_complex_to_log_grid(freq, mag_db, phase_deg, grid_n=2400, 
                                 fmin=FREQ_MIN_DEFAULT, fmax=FREQ_MAX_DEFAULT):
    f=np.asarray(freq); mdb=np.asarray(mag_db); ph=np.asarray(phase_deg)
    mask=(f>0); f,mdb,ph=f[mask],mdb[mask],ph[mask]
    if len(f)<4: raise RuntimeError("Not enough points in measurement.")
    xg=np.linspace(math.log2(fmin),math.log2(fmax),grid_n); fg=np.power(2.0,xg)
    x=np.log2(f); idx=np.argsort(x); x,mdb,ph=x[idx],mdb[idx],ph[idx]
    v=(xg>=x[0])&(xg<=x[-1])
    mdb_i=np.full_like(xg,np.nan,dtype=float); ph_i=np.full_like(xg,np.nan,dtype=float)
    mdb_i[v]=np.interp(xg[v],x,mdb)
    ph_un=unwrap_deg(ph); ph_i[v]=np.interp(xg[v],x,ph_un); ph_i[v]=wrap_phase_deg(ph_i[v])
    amp=np.full_like(xg,np.nan,dtype=float); amp[v]=np.power(10.0,mdb_i[v]/20.0)
    H=np.full_like(xg,np.nan+1j*np.nan,dtype=np.complex128); H[v]=amp[v]*np.exp(1j*np.deg2rad(ph_i[v]))
    return fg,H,v

# ---------- Crossovers ----------
def butterworth_poles(n):
    k=np.arange(1,n+1); ang=np.pi*(2*k+n-1)/(2*n); return np.exp(1j*ang)

def butterworth_lp_transfer(n,fc,freqs):
    if n<=0: return np.ones_like(freqs,dtype=np.complex128)
    w=2*np.pi*freqs; w0=2*np.pi*fc; jwn=1j*w/w0
    H=np.ones_like(jwn,dtype=np.complex128)
    for p in butterworth_poles(n): H*=1.0/(jwn-p)
    return H

def butterworth_hp_transfer(n,fc,freqs):
    if n<=0: return np.ones_like(freqs,dtype=np.complex128)
    w=2*np.pi*freqs; w0=2*np.pi*fc; jwn=1j*w/w0
    return (jwn**n)*butterworth_lp_transfer(n,fc,freqs)

def lr_transfer(kind,order,fc,freqs):
    if order%2!=0 or order<2 or order>8: raise ValueError("LR order must be even (2..8).")
    base=order//2
    Hb=butterworth_lp_transfer(base,fc,freqs) if kind=='lp' else butterworth_hp_transfer(base,fc,freqs)
    return Hb*Hb

def _parse_topology_to_order(t):
    t=t.strip().upper()
    if t=="NONE": return ("NONE",0)
    fam=t[:2]
    try: val=int(t[2:])
    except: return ("NONE",0)
    ordn=val//6 if val in (12,24,36,48) else val
    ordn=max(1,min(8,ordn))
    if fam=="LR" and ordn%2: ordn=ordn+1 if ordn<8 else 8
    return (fam,ordn)

def apply_filter(H,freqs,ftype,topology,fc):
    if H is None: return None
    if ftype=='none' or topology=='None' or fc is None or fc<=0: return H
    fam,ordn=_parse_topology_to_order(topology); kind=ftype.lower()
    if fam=="LR": Hf=lr_transfer(kind,ordn,fc,freqs)
    elif fam=="BW": 
        Hf=butterworth_lp_transfer(ordn,fc,freqs) if kind=='lp' else butterworth_hp_transfer(ordn,fc,freqs)
    else: Hf=np.ones_like(freqs,dtype=np.complex128)
    out=np.full_like(H,np.nan+1j*np.nan,dtype=np.complex128); ok=np.isfinite(H); out[ok]=H[ok]*Hf[ok]
    return out

# ---------- Smoothing ----------
# ---------- Smoothing ----------

def _smooth_segment(freqs_seg, y_seg, width_oct):
    """
    Smooth a contiguous non-NaN segment in log-f space.
    Edge handling: we *trim* the first/last half-window so the plot breaks
    cleanly (NaNs) instead of diving at segment boundaries.
    """
    x = np.log2(freqs_seg)
    n = len(x)
    if n < 3:
        return y_seg.copy()

    # resample to uniform log grid for a stable boxcar
    xg = np.linspace(x[0], x[-1], n)
    yg = np.interp(xg, x, y_seg)

    dx = xg[1] - xg[0] if n > 1 else 1.0
    L = max(3, int(round(width_oct / dx)))
    if L % 2 == 0:
        L += 1
    k = np.ones(L, dtype=float)

    # proper normalization near edges (no zero-padding bias)
    num = np.convolve(yg, k, mode='same')
    den = np.convolve(np.ones_like(yg), k, mode='same')
    ys = num / np.maximum(den, 1e-12)

    # trim edges that don't have a full window -> NaNs there
    he = L // 2
    start = he
    end = n - he
    if end - start < 2:
        # too short for a valid window; fall back to original (no smoothing)
        return y_seg.copy()

    out_seg = np.full_like(y_seg, np.nan, dtype=float)
    mask = (x >= xg[start]) & (x <= xg[end - 1])
    out_seg[mask] = np.interp(x[mask], xg[start:end], ys[start:end])
    return out_seg

def _nan_aware_smooth(freqs, y, frac_str, is_phase=False):
    if not frac_str or frac_str.lower() == 'none':
        return y
    try:
        N = float(frac_str.split('/')[1]) if '/' in frac_str else float(frac_str)
        width_oct = 1.0 / max(N, 1e-9)
    except:
        return y

    y = np.asarray(y, dtype=float)
    out = np.array(y, copy=True)
    finite = np.isfinite(y)
    if not np.any(finite):
        return out

    idx = np.where(finite)[0]
    starts = [idx[0]]; ends = []
    for i in range(1, len(idx)):
        if idx[i] != idx[i-1] + 1:
            ends.append(idx[i-1]); starts.append(idx[i])
    ends.append(idx[-1])

    for s, e in zip(starts, ends):
        seg = slice(s, e + 1)
        if e - s + 1 < 3:
            continue
        fseg = freqs[seg]
        yseg = out[seg]
        if is_phase:
            sm = _smooth_segment(fseg, unwrap_deg(yseg), width_oct)
            # wrap only finite points; NaNs stay NaN to break the line
            finite_sm = np.isfinite(sm)
            sm[finite_sm] = wrap_phase_deg(sm[finite_sm])
            out[seg] = sm
        else:
            out[seg] = _smooth_segment(fseg, yseg, width_oct)
    return out

def smooth_fractional_octave_mag(freqs, mag_db, frac_str):
    return _nan_aware_smooth(freqs, mag_db, frac_str, False)

def smooth_fractional_octave_phase(freqs, ph_wrapped, frac_str):
    return _nan_aware_smooth(freqs, ph_wrapped, frac_str, True)


# ---------- FIR EQ ----------
class PhaseBand:
    def __init__(self,f_hz=1000.0,Q=2.0,gain_deg=0.0,bypass=False):
        self.f_hz=f_hz; self.Q=Q; self.gain_deg=gain_deg; self.bypass=bypass

class MagBand:
    def __init__(self,f_hz=1000.0,Q=2.0,gain_db=0.0,bypass=False):
        self.f_hz=f_hz; self.Q=Q; self.gain_db=gain_db; self.bypass=bypass

def _phase_bell_deg(freqs,f0,Q,gain_deg):
    f0=max(1.0,float(f0)); Q=max(0.05,float(Q))
    logx=np.log2(np.maximum(freqs,1e-9)/f0); sigma_oct=0.5/Q
    bell=np.exp(-0.5*(logx/max(1e-6,sigma_oct))**2)
    phi=gain_deg*bell
    if len(phi)>0: phi[0]=0.0; phi[-1]=0.0
    return phi

def _mag_bell_db(freqs,f0,Q,gain_db):
    f0=max(1.0,float(f0)); Q=max(0.05,float(Q))
    logx=np.log2(np.maximum(freqs,1e-9)/f0); sigma_oct=0.5/Q
    bell=np.exp(-0.5*(logx/max(1e-6,sigma_oct))**2)
    return gain_db*bell

def build_phase_target(fs,n_fft,phase_bands):
    kmax=n_fft//2; freqs=np.linspace(0.0,fs/2.0,kmax+1)
    phi_deg=np.zeros_like(freqs,dtype=float)
    for b in phase_bands:
        if b.bypass or abs(b.gain_deg)<1e-9: continue
        phi_deg += _phase_bell_deg(freqs,b.f_hz,b.Q,b.gain_deg)
    phi_rad=np.deg2rad(phi_deg); H_half=np.exp(1j*phi_rad)
    H_half[0]=complex(np.real(H_half[0]),0.0)
    if n_fft%2==0: H_half[-1]=complex(np.real(H_half[-1]),0.0)
    return freqs,H_half

def build_magnitude_target(fs,n_fft,mag_bands):
    kmax=n_fft//2; freqs=np.linspace(0.0,fs/2.0,kmax+1)
    mag_db=np.zeros_like(freqs,dtype=float)
    for b in mag_bands:
        if b.bypass or abs(b.gain_db)<1e-9: continue
        mag_db += _mag_bell_db(freqs,b.f_hz,b.Q,b.gain_db)
    mag_lin = np.power(10.0, mag_db/20.0)
    return freqs, mag_lin

def half_to_full_spectrum(H_half):
    conj=np.conjugate(H_half[1:-1])[::-1] if len(H_half)>2 else np.array([],dtype=complex)
    return np.concatenate([H_half,conj])

def _symmetry_score(vec):
    if vec.size<4: return 0.0
    a=vec-np.mean(vec); b=a[::-1]
    denom=(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
    return float(np.dot(a,b)/denom)

def synthesize_impulse_from_Hhalf(H_half,taps,window='hann',centering='middle',custom_shift=0,normalize_dc=True):
    N=(len(H_half)-1)*2
    H_full=half_to_full_spectrum(H_half)
    h0=np.fft.ifft(H_full).real
    peak=int(np.argmax(np.abs(h0)))
    if centering=='start': target=0
    elif centering=='custom': target=int(custom_shift)
    else: target=N//2
    if centering=='closest':
        candidates=list(range(target-4,target+5))
        best_shift=0; best_score=-1e9
        for cand in candidates:
            shift=cand-peak; hc=np.roll(h0,shift)
            score=_symmetry_score(hc[max(0,cand-taps//2):min(N,cand+taps//2)])
            if score>best_score or (abs(score-best_score)<1e-9 and abs(np.argmax(np.abs(hc))-cand)<abs(best_shift)):
                best_score=score; best_shift=shift
        h=np.roll(h0,best_shift)
    else:
        h=np.roll(h0,target-peak)
    start=0 if centering=='start' else max(0,(N//2)-(taps//2)); end=start+taps
    seg=np.concatenate([h[start:],h[:end-N]]) if end>N else h[start:end]
    if window=='hann': seg*=np.hanning(taps)
    elif window=='blackman': seg*=np.blackman(taps)
    if normalize_dc:
        s=float(np.sum(seg))
        if abs(s)>1e-12: seg=seg/s
    return seg

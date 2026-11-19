"""Step 30: do raw radar moments at a station tell dust from no dust? (run where MetPy is available)

Decodes the Level II volumes fetched by 29_fetch_level2.py with metpy.io.Level2File, samples the lowest tilt (0.5 deg)
within 4 km of every ASOS station inside 8-120 km of the radar, and writes one row per station and volume:
reflectivity, correlation coefficient (RHO), differential reflectivity (ZDR), radial velocity. Paths at the top.
Result of 19 Nov 2026: results/level2_check/ and the README of results/val_2025_eventsplit.
"""
import sys, glob, re, datetime as dt, warnings
import numpy as np, pandas as pd
from metpy.io import Level2File
warnings.filterwarnings("ignore")
U="/mnt/user-data/uploads/DustAhead/data"
SITE={"KIWA":(33.2892,-111.6700),"KEPZ":(31.8731,-106.6980)}
st=pd.read_csv(f"{U}/labels/stations.csv",dtype={"sid":str})
def moments(sweep, name):
    az=[];rows=[];hdr=None
    for ray in sweep:
        m=ray[4].get(name)
        if m is None: continue
        hdr=m[0]; az.append(ray[0].az_angle); rows.append(np.asarray(m[1],float))
    if hdr is None: return None
    n=max(len(r) for r in rows); A=np.full((len(rows),n),np.nan)
    for k,r in enumerate(rows): A[k,:len(r)]=r
    rng=(hdr.first_gate+np.arange(n)*hdr.gate_width)
    return np.array(az),rng,A
def polar(site,lat,lon):
    la0,lo0=np.deg2rad(site); la,lo=np.deg2rad(lat),np.deg2rad(lon)
    dy=(la-la0)*6371; dx=(lo-lo0)*6371*np.cos(la0)
    return np.hypot(dx,dy),(np.degrees(np.arctan2(dx,dy))%360)
def sample(mom,r0,a0,rad=4.0):
    az,rng,A=mom
    daz=np.abs((az-a0+180)%360-180)
    dazlim=np.degrees(rad/max(r0,rad))
    ii=np.flatnonzero(daz<=dazlim); jj=np.flatnonzero(np.abs(rng-r0)<=rad)
    if not len(ii) or not len(jj): return np.array([])
    return A[np.ix_(ii,jj)].ravel()
rows=[]
for radar in SITE:
    files=sorted(glob.glob(f"{U}/raw/level2/{radar}/{radar}*_V06"))
    S=st.copy(); S["r"],S["az"]=polar(SITE[radar],S.lat.values,S.lon.values); S=S[(S.r>8)&(S.r<120)]
    for f in files:
        t=dt.datetime.strptime(re.search(r"(\d{8}_\d{6})",f).group(1),"%Y%m%d_%H%M%S")
        try: L=Level2File(f)
        except Exception as e: print("ERR",f[-28:],e); continue
        ref=rho=zdr=vel=None
        for sw in L.sweeps[:4]:
            if not sw: continue
            if abs(sw[0][0].el_angle-0.5)>0.35: continue
            if ref is None and moments(sw,b"RHO") is not None:
                ref,rho,zdr=moments(sw,b"REF"),moments(sw,b"RHO"),moments(sw,b"ZDR")
            if vel is None and moments(sw,b"VEL") is not None: vel=moments(sw,b"VEL")
        if ref is None: print("no dualpol sweep",f[-28:]); continue
        for s in S.itertuples():
            R=sample(ref,s.r,s.az); H=sample(rho,s.r,s.az); Z=sample(zdr,s.r,s.az); V=sample(vel,s.r,s.az) if vel is not None else np.array([])
            ok=np.isfinite(R)
            rows.append(dict(radar=radar,t=t,sid=s.sid,r=s.r,beam_km=s.r*np.sin(np.deg2rad(0.5))+s.r**2/(2*8494),
                 echo=float(ok.mean()) if len(R) else np.nan, ref_med=np.nanmedian(R) if ok.any() else np.nan, ref_max=np.nanmax(R) if ok.any() else np.nan,
                 rho_med=np.nanmedian(H) if np.isfinite(H).any() else np.nan, rho_lo=float((H[np.isfinite(H)]<0.8).mean()) if np.isfinite(H).any() else np.nan,
                 zdr_med=np.nanmedian(Z) if np.isfinite(Z).any() else np.nan, vel_p90=np.nanpercentile(np.abs(V),90) if np.isfinite(V).any() else np.nan))
        print(radar,t,len(S),"stations",flush=True)
D=pd.DataFrame(rows); D.to_csv("l2_station_samples.csv",index=False); print(D.shape)

import wradlib as wradlib
import matplotlib.pyplot as pl
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.colors import from_levels_and_colors
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')
try:
    get_ipython().magic("matplotlib inline")
except:
    pl.ion()
import numpy as np
import datetime as dt
from osgeo import osr
from osgeo import gdal
import wradlib as wrl
import datetime as dt
import numpy as np
from wradlib.io import read_generic_netcdf
from wradlib.util import get_wradlib_data_file
import os
from external import *
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import numpy as np
import scipy.io as sio
from pyhdf.SD import SD, SDC
def read_trmm_pyhdf(filename1, filename2):

    hdf=SD(filename1, SDC.READ)
    hdf_1 = SD(filename2, SDC.READ)
    lat = hdf.select('Latitude')
    lat = lat[:,:]
    lon = hdf.select('Longitude')
    lon = lon[:,:]
    year = hdf.select('Year')
    year = year[:]
    month = hdf.select('Month')
    month = month[:]
    dayofmonth = hdf.select('DayOfMonth')
    dayofmonth = dayofmonth[:]
    dayofyear = hdf.select('DayOfYear')
    dayofyear = dayofyear[:]
    hour = hdf.select('Hour')
    hour=hour[:]
    minute = hdf.select('Minute')
    minute=minute[:]
    second = hdf.select('Second')
    second = second[:]
    millisecond = hdf.select('MilliSecond')
    millisecond=millisecond[:]
    date_array = zip(year, month, dayofmonth,
                     hour, minute, second,
                     millisecond.astype(np.int32) * 1000)
    pr_time = np.array([dt.datetime(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in
             date_array])
    status = hdf.select('status')
    status = status[:,:]
    pflag = hdf.select('rainFlag') 
    pflag = pflag[:,:]
    ptype = hdf.select('rainType') 
    ptype = pflag[:,:]
    zbb = hdf.select('HBB')
    zbb = zbb[:,:].astype(np.float32)
    bbwidth = hdf.select('BBwidth')
    bbwidth = bbwidth[:,:].astype(np.float32)
    bbstatus = hdf.select('BBstatus')
    bbstatus = bbstatus[:,:]
    refl = hdf_1.select('correctZFactor')
    refl = refl[:,:].astype(np.float32)
    refl[refl == -8888.] = np.nan
    refl[refl == -9999.] = np.nan
    refl = refl / 100.
    quality = hdf_1.select('dataQuality')
    quality = quality[:]

    # Check for bad data
    if max(quality) != 0:
        raise ValueError('TRMM contains Bad Data')

    # Determine the dimensions
    ndim = refl.ndim
    if ndim != 3:
        raise ValueError('TRMM Dimensions do not match!'
                         'Needed 3, given {0}'.format(ndim))

    tmp = refl.shape
    nscan = tmp[0]
    nray = tmp[1]
    nbin = tmp[2]

    # Reverse direction along the beam
    # TODO: Why is this reversed?
    refl = refl[::-1]

    # Simplify the precipitation flag
    ipos = (pflag >= 10) & (pflag < 20)#check when not using equal to
    icer = (pflag == 20)
    pflag[ipos] = 1
    pflag[icer] = 2

    # Simplify the precipitation types
    istr = (ptype >= 100) & (ptype <= 200)
    icon = (ptype >= 200) & (ptype <= 300)
    ioth = (ptype >= 300)
    inone = (ptype == -88)
    imiss = (ptype == -99)
    ptype[istr] = 1 #type of rain stratiform 
    ptype[icon] = 2 #type of rain convective
    ptype[ioth] = 3 #type of rain other type
    ptype[inone] = 0
    ptype[imiss] = -1

    # Extract the surface type #modify
    sfc = np.zeros((nscan, nray), dtype=np.uint8)
    i0 = (status % 10 == 0)
    sfc[i0] = 0 #ocean
    i1 = ((status - 1) % 10 == 0)
    sfc[i1] = 1 #land
    i2 = ((status - 2) % 10 == 0)
    sfc[i2] = 2 #coastline
    i3 = ((status - 4) % 10 == 0)
    sfc[i3] = 4 #inland lake
    i9 = ((status - 9) % 10 == 0)
    sfc[i9] = 9 #inland
    
    # bright band detection status
    #bb_detection_status = np.zeros((nscan, nray), dtype=np.uint8)
    #bb_detection_status[bbstatus == -11] = 0
    #k_1 = (0 < (bbstatus/16)) & ((bbstatus/16) < 50)
    #bb_detection_status[k_1] = 1
    #k_2 = (50 < (bbstatus/16)) & ((bbstatus/16) < 109)
    #bb_detection_status[k_2] = 2
#     Extract 2A23 quality
#     TODO: Why is the `quality` variable overwritten?
#    quality = np.zeros((nscan, nray), dtype=np.uint8)
#    i0 = (status == 168)
#    quality[i0] = 0
    #i1 = ((bb_detection_status == 0) | (bb_detection_status == 1)) & (ptype == 1)
    #quality[i1] = 1
    #i2 = (bb_detection_status > 1) & (ptype >= 1)
    #quality[i2] = 2
    quality = np.zeros((nscan, nray), dtype=np.uint8)
    i0 = (status == 168)
    quality[i0] = 0
    i1 = (status < 50)
    quality[i1] = 1
    i2 = ((status >= 50) & (status < 109))
    quality[i2] = 2
    trmm_data = {}
    trmm_data.update({'nscan': nscan, 'nray': nray, 'nbin': nbin,
                      'date': pr_time, 'lon': lon, 'lat': lat,
                      'pflag': pflag, 'ptype': ptype, 'zbb': zbb,
                      'bbwidth': bbwidth, 'sfc': sfc, 'quality': quality,
                      'refl': refl, 'bbstatus': bbstatus})
    return trmm_data

def _get_tilts(dic):
    i = 0
    for k in dic.keys():
        if 'dataset' in k:
            i += 1
    return i

def read_gr(filename, loaddata=True):

    gr_data = wrl.io.read_generic_netcdf(filename)
    dat = gr_data['what']['date']
    tim = gr_data['what']['time']
    date = dt.datetime.strptime(dat + tim, "%Y%d%m%H%M%S")
    source = gr_data['what']['source']

    lon = gr_data['where']['lon']
    lat = gr_data['where']['lat']
    alt = gr_data['where']['height']

    if gr_data['what']['object'] == 'PVOL':
        ntilt = _get_tilts(gr_data)
    else:
        raise ValueError('GR file is no PPI/Volume File')

    ngate = np.zeros(ntilt, dtype=np.int16)
    nbeam = np.zeros(ntilt)
    elang = np.zeros(ntilt)
    r0 = np.zeros(ntilt)
    dr = np.zeros(ntilt)
    a0 = np.zeros(ntilt)

    for i in range(0, ntilt):
        dset = gr_data['dataset{0}'.format(i+1)]
        a0[i] = dset['how']['astart']
        elang[i] = dset['where']['elangle']
        ngate[i] = dset['where']['nbins']
        r0[i] = dset['where']['rstart']
        dr[i] = dset['where']['rscale']
        nbeam[i] = dset['where']['nrays']

    if ((len(np.unique(r0)) != 1) |
            (len(np.unique(dr)) != 1) |
            (len(np.unique(a0)) != 1) |
            (len(np.unique(nbeam)) != 1) |
            (nbeam[0] != 360)):
        raise ValueError('GroundRadar Data layout dos not match')

    gr_dict = {}
    gr_dict.update({'source': source, 'date': date, 'lon': lon, 'lat': lat,
                    'alt': alt, 'ngate': ngate, 'nbeam': nbeam, 'ntilt': ntilt,
                    'r0': r0, 'dr': dr, 'a0': a0, 'elang': elang})
    if not loaddata:
        return gr_dict

    sdate = []
    refl = []
    for i in range(0, ntilt):
        dset = gr_data['dataset{0}'.format(i+1)]
        dat = dset['what']['startdate']
        tim = dset['what']['starttime']
        date = dt.datetime.strptime(dat + tim, "%Y%d%m%H%M%S")
        sdate.append(date)
        data = dset['data1']
        quantity = data['what']['quantity']
        factor = data['what']['gain']
        offset = data['what']['offset']
        if quantity == 'DBZH':
            dat = data['variables']['data']['data'] * factor + offset
            refl.append(dat)

    sdate = np.array(sdate)
    refl = np.array(refl)

    gr_dict.update({'sdate': sdate, 'refl': refl})

    return gr_dict

# Set parameters for this procedure
bw_pr = 0.71                  # PR beam width
platf = "trmm"                 # PR platform/product: one out of ["gpm", "trmm_2a23", "trmm_2a25"]
zt = pr_pars[platf]["zt"]     # PR orbit height (meters)
dr_pr = pr_pars[platf]["dr"]  # PR gate length (meters)
ee = 2

# define GPM data set
#gpm_file = wradlib.util.get_wradlib_data_file('gpm/2A-RW-BRS.GPM.Ku.V6-20160118.20141206-S095002-E095137.004383.V04A.HDF5')

# define matching ground radar file
gr2gpm_file = wradlib.util.get_wradlib_data_file('hdf5/IDR66_20141206_094829.vol.h5')

# define TRMM data sets
trmm_2a23_file = wradlib.util.get_wradlib_data_file('trmm/2A-CS-151E24S154E30S.TRMM.PR.2A23.20100206-S111425-E111526.069662.7.HDF')
trmm_2a25_file = wradlib.util.get_wradlib_data_file('trmm/2A-CS-151E24S154E30S.TRMM.PR.2A25.20100206-S111425-E111526.069662.7.HDF')

# define matching ground radar file
gr2trmm_file = wradlib.util.get_wradlib_data_file('hdf5/IDR66_20100206_111233.vol.h5')
#alok_file_a=trmm_2a23_file[-11:-6]
#alok_file_b=trmm_2a23_file[-20:-11]
#alok_file_c=gr2trmm_file[-10:-2]
#alok_file_d=alok_file_c+alok_file_b+alok_file_a
#alok_file_coord='coordinates.'+alok_file_d+'.mat'
#alok_file_refl='reflectivity.'+alok_file_d+'.mat'
#alok_file_refl_text_file='reflectivity.'+alok_file_d+'.txt'
#overpass_orbit=['88616','88738','89028','89138','89486','89547','89608','89611','89669','89718']
#overpass_orbit_time=[1.0, -2.0, -3.0, -3.0, 0.0, -4.0, 0.0, -3.0, -5.0, -1.0]
#for ii in range(len(overpass_orbit)):
#     if (overpass_orbit[ii]==trmm_2a23_file[-11:-6]):
#             time_diff_for_orbit=overpass_orbit_time[ii]
#sweep_no=alok_file_c[-3:-1]
#sweep_no=float(sweep_no)
#if sweep_no>1.0:
#    time_diff_for_orbit=time_diff_for_orbit+(sweep_no-1)
#else:
#    time_diff_for_orbit=time_diff_for_orbit

# read matching GR data
if platf == "gpm":
    gr_data = read_gr2(gr2gpm_file)
elif platf=="trmm":
    gr_data = read_gr(gr2trmm_file)
else:
    raise("Invalid platform")

# number of rays in gr sweep
nray_gr = gr_data['nbeam'].astype("i4")[ee]
# number of gates in gr beam
ngate_gr = gr_data['ngate'].astype("i4")[ee]
# number of sweeps
nelev = gr_data['ntilt']
# elevation of sweep (degree)
elev = gr_data['elang'][ee]
# gate length (meters)
dr_gr = gr_data['dr'][ee]
# reflectivity array of sweep
ref_gr = gr_data['refl'][ee]
# sweep datetime stamp
date_gr = gr_data['sdate'][ee]
# range of first gate
r0_gr = gr_data['r0'][ee]
# azimuth angle of first beam
a0_gr = gr_data['a0'][ee]
# Longitude of GR
lon0_gr = gr_data['lon']
# Latitude of GR
lat0_gr = gr_data['lat']
# Altitude of GR (meters)
alt0_gr = gr_data['alt']
# Beam width of GR (degree)
bw_gr = 1.
print(elev, lon0_gr)

# read spaceborn PR data
if platf == "gpm":
    pr_data = read_gpm(gpm_file)
elif platf == "trmm":
    pr_data = read_trmm_pyhdf(trmm_2a23_file, trmm_2a25_file)
else:
    raise("Invalid platform")
refl = pr_data['refl']
#print(refl)
# Longitudes of PR scans
pr_lon = pr_data['lon']
# Latitudes of PR scans
pr_lat = pr_data['lat']
# Precip flag
pflag = pr_data['pflag']
# Number of scans on PR data
nscan_pr= pr_data['nscan']
# Number of rays in one PR scan
nray_pr = pr_data['nray']
# Number of gates in one PR ray
ngate_pr = pr_data['nbin']
# Precipiation type
precipitation_type = pr_data['ptype']
##
# Calculate equivalent earth radius
wgs84 = wradlib.georef.get_default_projection()
re1 = wradlib.georef.get_earth_radius(lat0_gr, wgs84) * 4./3.
print("eff. Earth radius 1:", re1)
a = wgs84.GetSemiMajor()
b = wgs84.GetSemiMinor()
print("SemiMajor, SemiMinor:", a, b)

# Set up aeqd-projection gr-centered
rad = wradlib.georef.proj4_to_osr(('+proj=aeqd +lon_0={lon:f} ' + 
                                   '+lat_0={lat:f} +a={a:f} ' +
                                   '+b={b:f}').format(lon=lon0_gr,
                                                      lat=lat0_gr,
                                                      a=a, b=b))
re2 = wradlib.georef.get_earth_radius(lat0_gr, rad) * 4./3.
print("eff. Earth radius 2:", re2)

# TODO: Seperate the insides of wradlib.georef.polar2lonlatalt_n 

# create gr range and azimuth arrays
rmax_gr = r0_gr + ngate_gr * dr_gr
r_gr = np.arange(0, ngate_gr) * dr_gr + dr_gr/2.
az_gr = np.arange(0, nray_gr) - a0_gr
print("Range/Azi-Shape:", r_gr.shape, az_gr.shape)

# create gr lonlat grid ##alok modified (check for lat lon grid)
gr_polargrid = np.meshgrid(r_gr, az_gr)
## alok modified (check for lat lon grid) gr_lon, gr_lat, gr_alt = wradlib.georef.polar2lonlatalt_n(gr_polargrid[0], gr_polargrid[1], elev, (lon0_gr, lat0_gr, alt0_gr ))
## alok modified (check for lat lon grid) gr_ll = np.dstack((gr_lon, gr_lat, gr_alt))
## alok modified (check for lat lon grid) print("LonLatAlt-Grid-Shape", gr_ll.shape)

# reproject to xyz
## alok modified (check for lat lon grid) gr_xyz = wradlib.georef.reproject(gr_ll, projection_source=wgs84, projection_target=rad)
gr_xyz,rad = wradlib.georef.spherical_to_xyz(gr_polargrid[0], gr_polargrid[1], elev, (lon0_gr, lat0_gr, alt0_gr )) ## alok modified (check for lat lon grid)
print("XYZ-Grid-Shape:", gr_xyz.shape)

# get radar domain (outer ring)
gr_domain = gr_xyz[:,-1,0:2]
gr_domain = np.vstack((gr_domain, gr_domain[0]))
print("Domain-Shape:", gr_domain.shape)

pr_x, pr_y = wradlib.georef.reproject(pr_lon, pr_lat, 
                                      projection_source=wgs84, 
                                      projection_target=rad)
pr_xy = np.dstack((pr_x, pr_y))
print("PR-GRID-Shapes:", pr_x.shape, pr_y.shape, pr_xy.shape)

# Create ZonalData for spatial subsetting (inside GR range domain)
## alok modified (14/1/2019) l_gr = []
## alok modified (14/1/2019) l_gr.append(gr_domain)
## alok modified (14/1/2019) zd = wradlib.zonalstats.ZonalDataPoint(pr_xy.reshape(-1, pr_xy.shape[-1]), l_gr, srs=rad, buf=500.)
## alok modified (14/1/2019) obj1 = wradlib.zonalstats.GridPointsToPoly(zd)

#    Get source indices within GR-Domain from zonal object
#    (0 because we have only one zone)
## alok modified (14/1/2019) pr_idx = obj1.zdata.get_source_index(0) 

# Subsetting in order to use only precipitating profiles
## alok modified (14/1/2019) src_idx = np.zeros_like(pflag, dtype=np.bool)
## alok modified (14/1/2019) mask = np.unravel_index(pr_idx, pflag.shape)
## alok modified (14/1/2019) src_idx[mask] = True

# get precip indexes
## alok modified (14/1/2019) precip_mask = (pflag == 2)
## alok modified (14/1/2019) precip_idx = src_idx & precip_mask
precip_mask = (pflag == 2) & wrl.zonalstats.get_clip_mask(pr_xy, gr_domain, rad)
## pl.imshow(precip_mask)
# get iscan/iray boolean arrays
## alok modified (14/1/2019) iscan = precip_idx.nonzero()[0]
## alok modified (14/1/2019) iray = precip_idx.nonzero()[1]
iscan = precip_mask.nonzero()[0]
iray = precip_mask.nonzero()[1]

print("NRAY", nray_pr)
print("NBIN", ngate_pr)

# Approximation!
alpha = abs(-17.04 + np.arange(nray_pr) * bw_pr)

# Correct for parallax, get 3D-XYZ-Array
#   xyzp_pr: Parallax corrected xyz coordinates
#   r_pr_inv: range array from ground to PR platform
#   zp: PR bin altitudes
xyzp_pr, r_pr_inv, z_pr = correct_parallax(pr_xy, nray_pr, ngate_pr, dr_pr, alpha)

print("PR_XYP:", xyzp_pr.shape, z_pr.shape)
#parallax corrected pr values(alok modified)
# TODO: Do we have to consider refraction in sat2pol?
r_pr, elev_pr, az_pr = sat2pol(xyzp_pr, (lon0_gr, lat0_gr, alt0_gr), re1)#done    
mask = (elev_pr > (1.0 - bw_gr/2.)) & (elev_pr < (1.0 + bw_gr/2.))#done
##pl.figure()
##pl.pcolormesh(mask[90,:,:].T)

# PR pulse volumes

# Range of PR bins
dists = dist_from_orbit(zt, alpha, r_pr_inv)#done

## Original IDL code...
##    rt=zt/COS(!dtor*alpha)-range
##    volp=(1.e-9)*!pi*(rt*!dtor*bwt/2.)^2*drt
## Translated to Python
vol_pr2  = np.pi * dr_pr * (dists * np.radians(bw_pr / 2.))**2
##fig = pl.figure(figsize=(12,4))
##pm = pl.pcolor(vol_pr.T)
##pl.colorbar(pm)

# Or using wradlib's native function
vol_pr = wradlib.qual.pulse_volume(dists, dr_pr, bw_pr)
#vol_pr = np.pi * dr_pr * (dists ** 2) * (np.tan(np.radians(bw_pr/2.))) ** 2

# Evaluate difference between both approaches
print("Min. difference (m3):", (vol_pr - vol_pr2).min())
print("Max. difference (m3): ", (vol_pr - vol_pr2).max())
print("Average rel. difference (%):", round(np.mean(vol_pr-vol_pr2)*100./np.mean(np.mean(vol_pr2)), 4))

# Verdict: differences are negligble - use wradlibs's native function!

# GR pulse volumes
#   along one beam
vol_gr = wradlib.qual.pulse_volume(r_gr, dr_gr, bw_gr)#done
#   with shape (nray_gr, ngate_gr)
vol_gr = np.repeat(vol_gr, nray_gr).reshape((nray_gr,ngate_gr), order="F")#done

ratio, zbb, median_bb_height, bb_width = get_bb_ratio(pr_data, z_pr) #bright band height and ratio
##pl.pcolormesh(ratio[60,:,:].T, vmin=-1, vmax=2)
##pl.colorbar()

# REVERSE!!!
refp = pr_data['refl'][:,:,::-1]
print("REFP:", refp.shape)

refp_ss = np.zeros_like(refp) * np.nan
refp_sh = np.zeros_like(refp) * np.nan

a_s, a_h = s_ku_coefficients()

ia = (ratio >= 1)
refp_ss[ia] = refp[ia] + calculate_polynomial(refp[ia], a_s[:,10])
refp_sh[ia] = refp[ia] + calculate_polynomial(refp[ia], a_h[:,10])
ib = (ratio <= 0)
refp_ss[ib] = refp[ib] + calculate_polynomial(refp[ib], a_s[:,0])
refp_sh[ib] = refp[ib] + calculate_polynomial(refp[ib], a_h[:,0])
im = (ratio > 0) & (ratio < 1)
ind = np.round(ratio[im] * 10).astype(np.int)
#print("W:", a_s[:,ind].shape)
refp_ss[im] = refp[im] + calculate_polynomial(refp[im], a_s[:,ind])
refp_sh[im] = refp[im] + calculate_polynomial(refp[im], a_h[:,ind])

refp_ss[refp < 0] = np.nan
out = np.ma.masked_invalid(refp_ss)
##pl.figure()
##pl.pcolormesh(out[60,:,:].T, vmin=0, vmax=60)
##pl.colorbar()
##pl.figure()
##pl.pcolormesh(refp[60,:,:].T, vmin=0, vmax=60)
##pl.colorbar()
##pl.figure()
##pl.pcolormesh(ratio[60,:,:].T, vmin=-1, vmax=2)
##pl.colorbar()

# Convert S-band GR reflectivities to Ku-band using method of Liao and Meneghini (2009)
ref_gr_ku = np.zeros_like(ref_gr) * np.nan

# Which zbb value should we take here???
#    Q'n'Dirty: just take the mean of all PR profiles
#    TODO: Consider zbb for each profile during the matching process

# Snow
ia = ( gr_xyz[...,2] >= np.nanmean(zbb) )
ref_gr_ku[ia] = ku2s["snow"][0] + ku2s["snow"][1]*ref_gr[ia] + ku2s["snow"][2]*ref_gr[ia]**2

# Rain
ib = ( gr_xyz[...,2] < np.nanmean(zbb) )
ref_gr_ku[ib] = ku2s["rain"][0] + ku2s["rain"][1]*ref_gr[ib] + ku2s["rain"][2]*ref_gr[ib]**2

# Jackson Tan's fix for C-band
is_cband = True
if (is_cband):
    delta = (ref_gr_ku - ref_gr) * 5.3/10.0
    ref_gr_ku = ref_gr + delta

# First assumption: no valid PR bins (all False)
valid = np.asarray(elev_pr, dtype=np.bool)==False
# PR is inside GR range and is precipitating
valid[iscan,iray] = True
# PR bins intersect with GR sweep
valid = valid & (elev_pr >= elev-bw_gr/2.) & (elev_pr <= elev+bw_gr/2.)
# Number of matching PR bins per profile
nvalids = np.sum(valid, axis=2)
# scan and ray indices for profiles with at least one valid bin
vscan, vray = np.where(nvalids>0)
# number of profiles with at least one valid bin
nprof = len(vscan)
# Lots of containers to store samples (only for one GR sweep angle!)
x = np.zeros(nprof)*np.nan        # x coordinate of sample
y = np.zeros(nprof)*np.nan        # y coordinate of sample
z = np.zeros(nprof)*np.nan        # z coordinate of sample
precip_type = np.zeros(nprof,dtype="i4")*np.nan #precipitation type
threshold_percentage_GR = np.zeros(nprof)*np.nan #threshold percentage for GR
threshold_percentage_PR = np.zeros(nprof)*np.nan #threshold percentage for PR
dz = np.zeros(nprof)*np.nan       # depth of sample
ds = np.zeros(nprof)*np.nan       # width of sample
rs = np.zeros(nprof)*np.nan       # range of sample from GR
refpr1 = np.zeros(nprof)*np.nan     # PR reflectivity
refpr2 = np.zeros(nprof)*np.nan     # PR reflectivity (S-band, snow)
refpr3 = np.zeros(nprof)*np.nan     # PR reflectivity (S-band, hail)  
refgr1 = np.zeros(nprof)*np.nan     # GR reflectivity
refgr2 = np.zeros(nprof)*np.nan     # GR reflectivity (Ku-band)
ntotpr = np.zeros(nprof,dtype="i4")# total number of PR bins in sample
nrej1 = np.zeros(nprof,dtype="i4")# number of rejected PR bins in sample
ntotgr = np.zeros(nprof,dtype="i4")# total number of GR bins in sample
nrej2 = np.zeros(nprof,dtype="i4")# number of rejected GR bins in sample
iref1 = np.zeros(nprof)*np.nan    # path-integrated PR reflectivity
iref2 = np.zeros(nprof)*np.nan    # path-integrated GR reflectivity
stdv1 = np.zeros(nprof)*np.nan    # std. dev. of PR reflectivity in sample
stdv2 = np.zeros(nprof)*np.nan    # std. dev. of GR reflectivity in sample
volpr = np.zeros(nprof)*np.nan     # total volume of PR bins in sample
volgr = np.zeros(nprof)*np.nan     # total volume of GR bins in sample
# Loop over relevant PR profiles
X_axis="X_axis"
Y_axis="Y_axis"
Z_axis="Z_axis"
GR_refl_bins="GR_bin"
PR_refl="PR_Ref"
sweep= ee
alok_file_refl_text_file='binned_reflectivity_'+ str(sweep) +'_.txt'
alok_file_refl_2='matched_reflectivity_data_sweep_'+ str(sweep) +'_.txt'
f = open(alok_file_refl_text_file, 'wb')
f.write("%-15s %-15s %-15s %-15s %-15s\n"%(X_axis, Y_axis, Z_axis, GR_refl_bins, PR_refl))
for ii, (ss, rr)  in enumerate(zip(vscan,vray)):
    # Index and count valid bins in each profile
    ip = np.where(valid[ss,rr])[0]#important#done
    numbins = len(ip)
    ntotpr[ii]=numbins
    if numbins == 0:
        continue
    # Compute the mean position of these bins
    x[ii]=np.mean(xyzp_pr[ss,rr,ip,0])#done
    y[ii]=np.mean(xyzp_pr[ss,rr,ip,1])#done
    z[ii]=np.mean(xyzp_pr[ss,rr,ip,2])#done
    precip_type[ii]=precipitation_type[ss,rr] #done #save it in matfiles	
    # Thickness of the layer
    dz[ii]=(numbins * dr_pr) * np.cos( np.radians(alpha[rr]) )#done

    # PR averaging volume
    volpr[ii]=np.sum(vol_pr2[rr,ip])#done

    # Note mean TRMM beam diameter
    ds[ii]=np.radians(bw_pr) * np.mean( ( (zt-z[ii]) / np.cos( np.radians(alpha[rr]) ) ) )#check in spyder

    # Note distance from radar
    s=np.sqrt(x[ii]**2+y[ii]**2)#done
    rs[ii]=(re2+z[ii]) * np.sin(s/re2) / np.cos(np.radians(elev))#done
    
    # This should not be required because we applied ZonalData
    ### Check that sample is within radar range
    ##if r[ii,jj]+ds[ii,jj]/2. gt rmax then continue

    ## THIS IS THE ORIGINAL IDL CODE - IS THIS A BUG???
    ##ref1[ii,jj]=MEAN(refp1,/nan)
    ##ref3[ii,jj]=MEAN(refp2,/nan)
    ##ref4[ii,jj]=MEAN(refp3,/nan)
    
    # Simple linear average of reflectivity 
    #   - we can become fancier in the next step
    # ATTENTION: NEED TO FLIP ARRAY
    PR_reflectivity_bins_considered_for_matching=np.shape(np.flipud(refp) [ss,rr,ip])
    PR_reflectivity_bins_considered_for_matching=float(PR_reflectivity_bins_considered_for_matching[0])
    PR_reflectivity_bins_considered_which_are_above_threshold_value=np.shape(np.where((np.flipud(refp) [ss,rr,ip])>18.0))
    PR_reflectivity_bins_considered_which_are_above_threshold_value=float(PR_reflectivity_bins_considered_which_are_above_threshold_value[1])
    PR_Threshold_percentage=(PR_reflectivity_bins_considered_which_are_above_threshold_value/PR_reflectivity_bins_considered_for_matching)*100
    threshold_percentage_PR[ii] = PR_Threshold_percentage
    #####
    refpr1[ii]=10*np.log10(np.nanmean(10**(np.flipud(refp)[ss,rr,ip]/10)))   #donot forget to flip #done #mean of the reflectivity taken here #important
    refpr2[ii]=10*np.log10(np.nanmean(10**(np.flipud(refp_ss)[ss,rr,ip]/10)))
    refpr3[ii]=10*np.log10(np.nanmean(10**(np.flipud(refp_sh)[ss,rr,ip]/10)))
    
    ## Not sure why we need this...
    ### Note the number of rejected bins
    ##nrej1[ii,jj]=ROUND(TOTAL(FINITE(refp1,/nan)))
    ##if FINITE(stdv1[ii,jj]) eq 0 and np-nrej1[ii,jj] gt 1 then STOP

    # SHOULD WE USE ZONALDATA INSTEAD? COULD BE MORE ACCURATE, BUT ALSO SLOWER
    # WE COULD BASICALLY START A NEW LOOP HERE AND RUN ZONALDATA BEFORE
    
    # Compute the horizontal distance to all the GR bins
    d = np.sqrt((gr_xyz[...,0]-x[ii])**2 + (gr_xyz[...,1]-y[ii])**2)

    # Find all GR bins within the SR beam
    aa, bb = np.where(d <= ds[ii]/2.)

    # Store the number of bins
    ntotgr[ii] = len(aa)

    if len(aa) == 0:
        continue

    # Extract the relevant GR bins

    # Compute the GR averaging volume
    volgr[ii]=np.sum(vol_gr[aa,bb])

    # Average over those bins that exceed the reflectivity threshold 
    #   IDL code does exponential distance and volume weighting
    #   Let's try simple mean first,
    #   THEN ZonalStats!

    #print('GR refl shape:',np.shape(ref_gr[aa,bb]))
    GR_reflectivity_bins_considered_for_matching=np.shape(ref_gr[aa,bb])
    GR_reflectivity_bins_considered_for_matching=float(GR_reflectivity_bins_considered_for_matching[0])
    GR_reflectivity_bins_considered_which_are_above_threshold_value=np.shape(np.where(ref_gr[aa,bb].data>=0.0))
    GR_reflectivity_bins_considered_which_are_above_threshold_value=float(GR_reflectivity_bins_considered_which_are_above_threshold_value[1])
    GR_Threshold_percentage=(GR_reflectivity_bins_considered_which_are_above_threshold_value/GR_reflectivity_bins_considered_for_matching)*100
    threshold_percentage_GR[ii] = GR_Threshold_percentage
    #print('GR threshold percentage:',GR_Threshold_percentage)
    ####
    refgr1[ii]=10*np.log10(np.nanmean(10**(ref_gr[aa,bb]/10)))
    refgr2[ii]=10*np.log10(np.nanmean(10**(ref_gr_ku[aa,bb]/10)))
    GR_1=np.zeros(numbins)
    GR_2=np.zeros(numbins)
    if (np.sum(np.isnan(np.flipud(refp)[ss,rr,ip]))>0):
        GR_1[:]=np.nan
        GR_2[:]=np.nan
    elif (np.sum(np.abs(np.flipud(refp)[ss,rr,ip]))==0):
        GR_1[:]=refgr1[ii]/numbins
        GR_2[:]=refgr2[ii]/numbins
    elif (np.sum(np.abs(np.flipud(refp)[ss,rr,ip]))>0):
        for metrs in range(numbins):
            GR_1[metrs]=10*np.log10(10**(refgr1[ii]/10)*(10**(np.flipud(refp)[ss,rr,ip[metrs]]/10))/(np.sum(10**(np.flipud(refp)[ss,rr,ip]/10))))
            GR_2[metrs]=10*np.log10(10**(refgr2[ii]/10)*(10**(np.flipud(refp)[ss,rr,ip[metrs]]/10))/(np.sum(10**(np.flipud(refp)[ss,rr,ip]/10))))
    for m in range(numbins):
        k=ip[m]
        #print('x:',xyzp_pr[ss,rr,ip[m],0],'y:',xyzp_pr[ss,rr,ip[m],1],'z:',xyzp_pr[ss,rr,ip[m],2],'gr refl:',refgr2[ii],'refl pr:',np.flipud(refp)[ss,rr,ip[m]])
        #print("%-15f %-15f %-15f %-15f %-15f"%(xyzp_pr[ss,rr,ip[m],0], xyzp_pr[ss,rr,ip[m],1], xyzp_pr[ss,rr,ip[m],2], refgr2[ii], np.flipud(refp)[ss,rr,ip[m]]))
        f.write("%-15f %-15f %-15f %-15f %-15f\n"%(x[ii], y[ii], xyzp_pr[ss,rr,ip[m],2], GR_2[m], np.flipud(refp)[ss,rr,ip[m]]))
    	#print("beam diameter:",ds[ii])
f.close()  
fig = pl.figure(figsize=(12,5))
ax = fig.add_subplot(121, aspect="equal")
pl.scatter(refgr1, refpr1, marker="+", c="black")
pl.plot([0,60],[0,60], linestyle="solid", color="black")
pl.xlim(10,50)
pl.ylim(10,50)
pl.xlabel("GR reflectivity (dBZ)")
pl.ylabel("PR reflectivity (dBZ)")
ax = fig.add_subplot(122)
pl.hist(refgr1[refpr1>-10], bins=np.arange(-10,50,5), edgecolor="None", label="GR")
pl.hist(refpr1[refpr1>-10], bins=np.arange(-10,50,5), edgecolor="red", facecolor="None", label="PR")
pl.xlabel("Reflectivity (dBZ)")
pl.legend()
fig = pl.figure(figsize=(12,12))
ax = fig.add_subplot(121, aspect="equal")
pl.scatter(x, y, c=refpr1, cmap=pl.cm.jet, vmin=0, vmax=50, edgecolor="None")
pl.title("PR reflectivity")
pl.grid()
ax = fig.add_subplot(122, aspect="equal")
pl.scatter(x, y, c=refgr1, cmap=pl.cm.jet, vmin=0, vmax=50, edgecolor="None")
pl.title("GR reflectivity")
pl.grid()
mat_file_data={}
mat_file_data.update({'x': x, 'y': y, 'z': z, 'refl_pr': refpr1,
                      'ref_gr': refgr1,'ref_gr_ku':refgr2, 'precipitation_type':precip_type, 
		      'Threshold_Percentage_GR':threshold_percentage_GR,
		      'Threshold_Percentage_PR':threshold_percentage_PR#'All_PR_coords_axis':xyzp_pr,
		      #'Average_BB_Height':median_bb_height, 'Average_BB_Width':bb_width,
		      #'Overpass_time_diff':abs(time_diff_for_orbit)
		      })
#sio.savemat(alok_file_coord, {'x':np.x},{'y':np.y})
sio.savemat(alok_file_refl_2, mat_file_data)

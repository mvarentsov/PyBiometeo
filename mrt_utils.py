import numpy as np

import suncalc

from mrt_utils import * 
from tqdm import tqdm


BOLZMAN = 5.6704 * (10**-8)


def calc_LW_rad (t, emis):
    return BOLZMAN * emis * t**4

def solarvf (solar_elev):
    return 0.308*np.cos(np.radians(solar_elev * (1-solar_elev**2/48402)))

def calc_MRT (solar_elev, S_sky_ghi, S_sky_dhi, S_ground, S_walls, L_sky, L_ground, L_walls, psi_sky = 1, human_alb = 0.3, human_emis=0.97):

    human_absorb_sw = 1-human_alb 
    human_absorb_lw = human_emis

    solar_zenith = 90 - solar_elev

    psi_walls = 1 - psi_sky

    S_sky_dni = (S_sky_ghi - S_sky_dhi) / np.cos(np.radians(solar_zenith)) 

    sum_sw = 0.5 * (S_sky_dhi * psi_sky + S_walls * psi_walls + S_ground) + S_sky_dni * solarvf (solar_elev)
    sum_lw = 0.5 * (L_sky * psi_sky + L_walls * psi_walls + L_ground)

    Tmrt = ((human_absorb_sw * sum_sw + human_absorb_lw * sum_lw) / (human_absorb_lw * BOLZMAN))**0.25

    return Tmrt


def calc_MRT4surface (solar_elev, S_sky_ghi, S_sky_dhi, L_sky, t_g, ground_alb, ground_emis):
    return calc_MRT (solar_elev = solar_elev, 
                     S_sky_ghi = S_sky_ghi, 
                     S_sky_dhi = S_sky_dhi,
                     S_ground  = S_sky_ghi * ground_alb, 
                     S_walls   = 0,
                     L_sky     = L_sky,
                     L_ground  = calc_LW_rad(t_g, ground_emis),
                     L_walls = 0)

def calc_MRT4ds (ds, t2m_var, ghi_var, dhi_var, dlr_var, t_g_var, ground_alb, ground_emis, load4time=False):
    
    mrt_sun = ds[t2m_var] * np.nan
    mrt_shd = ds[t2m_var] * np.nan

    ground_alb  = 0.2
    ground_emis = 0.95

    for i,time in tqdm (enumerate(ds.time), total=ds.time.shape[0]):

        solar_pos = suncalc.get_position(time.values, ds['lon'].mean().values, ds['lat'].mean().values)
        solar_elev = np.degrees(solar_pos['altitude'])

        ds4time = ds.sel(time=time)

        if load4time:
            ds4time.load()

        mrt_sun[i,:,:] = calc_MRT4surface (solar_elev, ds4time[ghi_var], ds4time[dhi_var], ds4time[dlr_var], ds4time[t_g_var], ground_alb, ground_emis)
        mrt_shd[i,:,:] = calc_MRT4surface (solar_elev, ds4time[dhi_var], ds4time[dhi_var], ds4time[dlr_var], ds4time[t_g_var], ground_alb, ground_emis)

    return mrt_sun, mrt_shd
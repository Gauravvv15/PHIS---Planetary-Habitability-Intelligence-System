import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.preprocessing import LabelEncoder


def clean_name(name):
    if pd.isna(name):
        return ''
    return name.strip().lower().replace(' ','')

def preprocessing_train(nasa_df, phl_df):
    nasa_df['key_name']=nasa_df['pl_name'].apply(clean_name)
    phl_df['key_name']=phl_df['P_NAME'].apply(clean_name)

    nasa_labels=nasa_df[[
        'key_name',
        #planatary important
        'pl_rade', 'pl_radj', 'pl_masse', 'pl_massj','pl_bmasse', 'pl_dens',  

        # orbital features
        'pl_orbper', 'pl_orbsmax', 'pl_orbeccen',

        #habitability features
        'pl_eqt', 'pl_insol',

        #stellar features
        'st_teff', 'st_rad', 'st_mass', 'st_lum', 'st_met', 'st_logg', 'st_age',

        #system features
        'sy_snum', 'sy_pnum', 'sy_dist',

        #discovery features
        'discoverymethod', 'disc_year',

        # id column 
        'pl_name', 'hostname'
    ]].copy()


    phl_df['P_HABITABLE']=phl_df['P_HABITABLE'].apply(
        lambda x:1 if x in [1, 2] else 0
)

    phl_features = phl_df[[
        'key_name', 'P_NAME',

        # ── TARGETS ───────────────────────────────────────────
        'P_HABITABLE', 'P_ESI', 'P_HABZONE_CON', 'P_HABZONE_OPT', 'P_TYPE',                

        # ── PLANET PHYSICAL (PHL measurements) ────────────────
        'P_MASS', 'P_RADIUS', 'P_DENSITY', 'P_GRAVITY', 'P_ESCAPE', 'P_GEO_ALBEDO', 'P_TEMP_MEASURED', 'P_TEMP_EQUIL', 'P_ATMOSPHERE',      

        # ── ORBITAL (PHL measurements) ────────────────────────
        'P_PERIOD', 'P_SEMI_MAJOR_AXIS', 'P_ECCENTRICITY', 'P_FLUX', 'P_FLUX_MIN', 'P_FLUX_MAX', 'P_DISTANCE_EFF', 'P_PERIASTRON', 'P_APASTRON', 'P_HILL_SPHERE', 'P_POTENTIAL', 

        # ── STELLAR (PHL measurements) ────────────────────────
        'S_TEMPERATURE', 'S_MASS', 'S_RADIUS', 'S_LUMINOSITY',  'S_METALLICITY', 'S_AGE', 'S_LOG_G', 

        # ── HABITABLE ZONE BOUNDARIES (PHL exclusive) ─────────
        'S_HZ_CON_MIN', 'S_HZ_CON_MAX', 'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX', 'S_HZ_CON0_MIN', 'S_HZ_CON0_MAX', 

        # ── STELLAR EXTRAS (PHL exclusive) ────────────────────
        'S_ABIO_ZONE', 'S_TIDAL_LOCK', 'S_TYPE', 

        # ── FILL HELPERS (used to patch NASA gaps) ────────────
        'P_MASS_EST', 'P_RADIUS_EST', 'P_SEMI_MAJOR_AXIS_EST',

    ]].copy()


    merged_df=phl_features.merge(nasa_df, on='key_name', how='left')



    merged_df['final_radius']= merged_df['pl_rade'].fillna(merged_df['P_RADIUS'])\
                                                    .fillna(merged_df['P_RADIUS_EST'])

    merged_df['final_mass']= merged_df['pl_masse'].fillna(merged_df['pl_bmasse'])\
                                                    .fillna(merged_df['P_MASS'])\
                                                    .fillna(merged_df['P_MASS_EST'])

    # Orbital period
    merged_df['final_period']=merged_df['pl_orbper'].fillna(merged_df['P_PERIOD'])

    # Semi-major axis
    merged_df['final_au_dist']= merged_df['pl_orbsmax'].fillna(merged_df['P_SEMI_MAJOR_AXIS'])\
                                                        .fillna(merged_df['P_SEMI_MAJOR_AXIS_EST'])

    # Eccentricity
    merged_df['final_ecc'] = merged_df['pl_orbeccen'].fillna(merged_df['P_ECCENTRICITY'])\
                                                        .fillna(0.0)  # circular orbit default

    # Equilibrium temperature
    merged_df['final_temp']= merged_df['pl_eqt'].fillna(merged_df['P_TEMP_EQUIL'])

    # Insolation flux
    merged_df['final_flux']= merged_df['pl_insol'].fillna(merged_df['P_FLUX'])

    # Stellar temperature
    merged_df['final_st_temp']= merged_df['st_teff'].fillna(merged_df['S_TEMPERATURE'])

    # Stellar mass
    merged_df['final_st_mass']= merged_df['st_mass'].fillna(merged_df['S_MASS'])

    # Stellar radius
    merged_df['final_st_rad']= merged_df['st_rad'].fillna(merged_df['S_RADIUS'])

    # Stellar luminosity
    merged_df['final_st_lum']= merged_df['st_lum'].fillna(merged_df['S_LUMINOSITY'])

    # Stellar metallicity
    merged_df['final_st_met']= merged_df['st_met'].fillna(merged_df['S_METALLICITY'])\
                                                    .fillna(0.0)  # solar metallicity default

    # Stellar surface gravity
    merged_df['final_st_logg']= merged_df['st_logg'].fillna(merged_df['S_LOG_G'])

    # Stellar age
    merged_df['final_st_age'] = merged_df['st_age'].fillna(merged_df['S_AGE'])




    merged_df['habitable']=merged_df['P_HABITABLE'].astype(int)   #for classification

    merged_df['esi']=merged_df['P_ESI'].fillna(0.0).astype(float) # for logistic regression 

    #habitable zone flag
    merged_df['hab_zone_con']=merged_df['P_HABZONE_CON'].fillna(0).astype(int)  #"Best Bet"—high probability of right temperature.
    merged_df['hab_zone_opt']=merged_df['P_HABZONE_OPT'].fillna(0).astype(int)  #"Long Shot"—could be habitable if the atmosphere is perfect.


    final_features=[
        #planets features
        'final_radius',
        'final_mass',
        'final_period',
        'final_ecc',
        'final_au_dist', 
        'final_temp',
        'final_flux',

        #phl planets
        'P_DISTANCE_EFF',       # effective distance (accounts for eccentricity)
        'P_PERIASTRON',         # closest point to star
        'P_APASTRON',           # farthest point from star

        #steallar features
        'final_st_temp',
        'final_st_mass',
        'final_st_rad',
        'final_st_lum',
        'final_st_logg',
        'final_st_met',
        'final_st_age',
        'S_ABIO_ZONE',          # abiogenesis zone flag
        'S_TIDAL_LOCK',         # tidal lock radius

        # HZ BOUNDARIES (PHL exclusive, huge signal)
        'S_HZ_CON_MIN',         # inner conservative HZ boundary
        'S_HZ_CON_MAX',         # outer conservative HZ boundary
        'S_HZ_OPT_MIN',         # inner optimistic HZ boundary
        'S_HZ_OPT_MAX',         # outer optimistic HZ boundary

        
        #SYSTEM (NASA)
        'sy_snum',             #binary stars destabilize HZ
        'sy_pnum',             #multi-planet stability
        'disc_year',
        'discoverymethod',   
        
        #targets and habitability context
        'esi',
        'habitable',
        'P_TYPE'
    ]

    df=merged_df[final_features].copy()

    #filling na's
    median_impute_cols=[
        'S_ABIO_ZONE','S_TIDAL_LOCK','final_st_age','S_HZ_OPT_MAX','S_HZ_OPT_MIN','S_HZ_CON_MAX','S_HZ_CON_MIN',
        'final_flux','final_st_logg','final_period','final_st_rad','final_st_lum','P_PERIASTRON','P_DISTANCE_EFF','P_APASTRON',
        'sy_dist','disc_year','final_st_mass','final_st_temp','final_au_dist',
        'final_sma', 'final_temp',
        'final_st_teff', 'final_radius', 'final_mass'
    ]

    for col in median_impute_cols:
        if col in df.columns:
            median_val=df[col].median()
            df[col]=df[col].fillna(median_val)

    df['sy_snum']=df['sy_snum'].fillna(1)
    df['sy_pnum']=df['sy_pnum'].fillna(1)


    #label encoding

    le_disc=LabelEncoder()
    df['discoverymethod_encoded']=le_disc.fit_transform(df['discoverymethod'].fillna('Unknown'))

    le_ptype=LabelEncoder()
    df['P_TYPE_encoded']=le_ptype.fit_transform(df['P_TYPE'].fillna('Unknown'))

    # le_ptpye_temp=LabelEncoder()
    # df['P_TYPE_TEMP_encoded']=le_ptpye_temp.fit_transform(df['P_TYPE_TEMP'].fillna('Unknown'))

    df=df.drop(['discoverymethod','P_TYPE'], axis=1)
  
    joblib.dump(le_disc, 'models/le_disc.pkl')
    joblib.dump(le_ptype, 'models/le_ptype.pkl')
 

    return df

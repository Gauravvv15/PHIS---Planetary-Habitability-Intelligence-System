import pandas as pd
import joblib
import numpy as np

le_disc=joblib.load('models/le_disc.pkl')
le_ptype=joblib.load('models/le_ptype.pkl')
df_cols=joblib.load('models/df_cols.pkl')
scaler=joblib.load('models/scaler.pkl')


feature_cols=[
    'final_radius', 'final_mass', 'final_period', 'final_ecc',
    'final_au_dist', 'final_temp', 'final_flux',
    'P_DISTANCE_EFF', 'P_PERIASTRON', 'P_APASTRON',
    'final_st_temp', 'final_st_mass', 'final_st_rad', 'final_st_lum',
    'final_st_logg', 'final_st_met', 'final_st_age',
    'S_ABIO_ZONE', 'S_TIDAL_LOCK',
    'S_HZ_CON_MIN', 'S_HZ_CON_MAX', 'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX',
    'sy_snum', 'sy_pnum', 'disc_year',
    'discoverymethod_encoded' ]

# def features_engineering(
#         planet_radius, #in earth radius
#         planet_mass,   #in earth mass
#         au_dist,       #AU from star
#         eccentricity,  #0.0 to 1.0
#         st_temp,       #in kelvin
#         st_mass,       #solar masses 
#         st_luminosity, #solar luminosity(linear)
#         st_metallicity, #[FE/H] - use 5.0 if unknown
#         st_age,        #in billion year eg(4.25 or 5.0)
#         st_radius      #solar radius
# ):
    

def features_engineering(input_df):
    #1 T(years) =sqrt(a3 / mass of the star) * 325.25
    input_df['final_period']=np.sqrt(input_df['au_dist']**3/input_df['st_mass']) * 365.25

    #2 stellar flux: Luminosity / AU_dist^2 (earth =1.0 at 1 AU from sun) 
    input_df['final_flux']=input_df['st_luminosity'] / (input_df['au_dist']**2)

    #3 Equilibrium temperature: star_temp * sqrt(radius of star / 2 AU_dist) * (1 - albedo)^0.25
    albedo=0.3
    input_df['st_radius_au']=input_df['st_radius'] * 0.00465047 #convert solar radius to AU
    input_df['final_temp']=input_df['st_temp'] * (np.sqrt(input_df['st_radius_au'] / (input_df['au_dist'] *2)) * (1 - albedo)**0.25)

    input_df['P_PERIASTRON']=input_df['au_dist'] * (1 - input_df['eccentricity']) #closest point to star
    input_df['P_APASTRON']=input_df['au_dist'] * (1 + input_df['eccentricity'])   #farthest point from star
    input_df['P_DISTANCE_EFF']=np.sqrt(1 - input_df['eccentricity'] ** 2)  

    #stallar log g: log g= log_g_Sun + log(M/M_sun) - 2Xlog(R/R_sun)
    # solar log g= 4.44
    input_df['final_st_log']= 4.44 + np.log10(input_df['st_mass']) - 2* np.log10(input_df['st_radius'])

    input_df['S_HZ_CON_MIN']=np.sqrt(input_df['st_luminosity'] / 1.1) #inner conservative HZ boundary
    input_df['S_HZ_CON_MAX']=np.sqrt(input_df['st_luminosity'] / 0.36) #outer conservative HZ boundary
    input_df['S_HZ_OPT_MIN']=np.sqrt(input_df['st_luminosity'] / 1.77) #inner optimistic HZ boundary
    input_df['S_HZ_OPT_MAX']=np.sqrt(input_df['st_luminosity'] / 0.29) #outer optimistic HZ boundary 

    lower_bound=0.77 * np.sqrt(input_df['st_luminosity'])
    upper_bound=1.02 * np.sqrt(input_df['st_luminosity'])

    input_df['S_ABIO_ZONE']=((lower_bound<= input_df['au_dist']) & (input_df['au_dist'] <= upper_bound)).astype(float)

    input_df['S_TIDAL_LOCK']= 0.5 * (input_df['st_mass'] ** (1/3)) #planet inside this radius is likely tidally locked

    #planet type
    radius=input_df['planet_radius'].iloc[0]
    if radius <=1.5: input_df['P_TYPE']='Terran'
    elif radius <= 2.5: input_df['P_TYPE']='SuperTerran'
    elif radius <= 6.0: input_df['P_TYPE']= 'Neptunian'
    else: input_df['P_TYPE']='Jovian'

    input_df['discovery_methods']= 'Transit' #most common method

    def safe_encoding(encoder, value):
        try :
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                #fallback to most common class
                return -1

        except Exception as e:
            print(f'Error encoding value {value} with encoder {encoder}: {e}')
            return -1
        

    input_df['discoverymethod_encoded']=safe_encoding(le_disc, input_df['discovery_methods'].iloc[0])
    input_df['ptype_encoded']=safe_encoding(le_ptype, input_df['P_TYPE'].iloc[0])

    input_df=input_df.drop(['discovery_methods','P_TYPE'], axis=1)



    input_df.rename(columns = {
        'planet_radius': 'final_radius',
        'planet_mass'  : 'final_mass',
        'final_period': 'final_period',
        'eccentricity'   : 'final_ecc',
        'au_dist': 'final_au_dist',
        'final_temp'  : 'final_temp',
        'final_flux'  : 'final_flux',
        'P_DISTANCE_EFF': 'P_DISTANCE_EFF',
        'P_PERIASTRON' : 'P_PERIASTRON',
        'P_APASTRON'  : 'P_APASTRON',
        'st_temp': 'final_st_temp',
        'st_mass': 'final_st_mass',
        'st_radius' : 'final_st_rad',
        'st_luminosity' : 'final_st_lum',
        'final_st_log': 'final_st_logg',
        'st_metallicity' : 'final_st_met',
        'st_age' : 'final_st_age',
        'S_ABIO_ZONE'  : 'S_ABIO_ZONE',
        'S_TIDAL_LOCK' : 'S_TIDAL_LOCK',
        'S_HZ_CON_MIN' : 'S_HZ_CON_MIN',
        'S_HZ_CON_MAX' : 'S_HZ_CON_MAX',
        'S_HZ_OPT_MIN' : 'S_HZ_OPT_MIN',
        'S_HZ_OPT_MAX' : 'S_HZ_OPT_MAX',
        'discoverymethod_encoded': 'discoverymethod_encoded',
        'ptype_encoded': 'P_TYPE_encoded'
    }, inplace=True)
    
    input_df['sy_snum'] = 1
    input_df['sy_pnum'] = 1
    input_df['disc_year'] = 2026

    final_features = [
    'final_radius', 'final_mass', 'final_period', 'final_ecc',
    'final_au_dist', 'final_temp', 'final_flux',
    'P_DISTANCE_EFF', 'P_PERIASTRON', 'P_APASTRON',
    'final_st_temp', 'final_st_mass', 'final_st_rad',
    'final_st_lum', 'final_st_logg', 'final_st_met',
    'final_st_age', 'S_ABIO_ZONE', 'S_TIDAL_LOCK',
    'S_HZ_CON_MIN', 'S_HZ_CON_MAX',
    'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX',
    'sy_snum', 'sy_pnum', 'disc_year',
    'discoverymethod_encoded', 'P_TYPE_encoded'
    ]

    for col in final_features:
        if col not in input_df.columns:
            input_df[col]=0


    numerical_cols = input_df.select_dtypes(include=['number']).columns

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    return input_df
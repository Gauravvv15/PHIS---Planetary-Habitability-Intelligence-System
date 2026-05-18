import pandas as pd
import numpy as np
from source.preprocessing_in import features_engineering
from source.predict import models_calculate


def run():
    def safefloat(value):
        while True:
            try:
                return float(input(value))
            except ValueError:
                print('Invalid Input, please input a valid value!')
                
    while True:
        user=input('Do you want to continue:')

        if user.lower() in ['no', 'n', 'stop', 'exit']:
            break
        
        else:
            planet_radius=safefloat('Enter planet radius in earth radius:')
            planet_mass=safefloat('Enter planet mass in earth mass:')
            au_dist=safefloat('Enter planet distance from star in Astronomical Units(AU):')
            eccentricity=safefloat('Enter Planet eccentricity (0.0 to 1.0):')
            st_temp=safefloat('Enter star temperature in kelvin:')
            st_mass=safefloat('Enter star mass in solar masses:')
            st_luminosity=safefloat('Enter star luminosity in solar Luminosity:')
            st_metallicity=safefloat('Enter star metallicity (0.0 to 5.0):')
            st_age=safefloat('Enter Star age in billions of years:')
            st_radius=safefloat('Enter star radius in solar radius:')

            input_df=pd.DataFrame({
                'planet_radius': [planet_radius],
                'planet_mass': [planet_mass],
                'au_dist': [au_dist],
                'eccentricity': [eccentricity],
                'st_temp': [st_temp],
                'st_mass': [st_mass],
                'st_luminosity': [st_luminosity],
                'st_metallicity': [st_metallicity],
                'st_age': [st_age],
                'st_radius': [st_radius]
            })


            in_df=features_engineering(input_df)
            models_calculate(in_df)

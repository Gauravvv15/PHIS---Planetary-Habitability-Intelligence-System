import joblib
import numpy as np

habitability_model = joblib.load('models/xgb_habitability_model.pkl')
esi_model= joblib.load('models/xgb_esi_model.pkl')

def models_calculate(df):
    habitability= predict_habitability(df)
    calculated_esi=predict_esi(df)

    print(habitability)
    print('\n')
    print(calculated_esi)


def predict_habitability(X):
    habitability_proba=habitability_model.predict_proba(X)[0]
    index=np.argmax(habitability_proba)
    confidence=habitability_proba[index]

    hab_pred=habitability_model.predict(X)
    if hab_pred == 0:
        hab_class='NOT HABITABLE'

        interpretation = f'''
        Planet Classification : {hab_class}
        Confidence= {confidence:.2%}
        Interpretation: 
        This Exo-Planet predicted to have low potential for supporting Earth-Like life based on its planetary physics, orbital conditions, stellar environment, and thermal properties.
        
        Possible contributing factors may include:
        - Extreme temperature conditions
        - High or low stellar flux
        - Unfavorable orbital distance
        - Poor planetary composition
        - Host star instability
        - Non-optimal habitability zone placement
        
        Scientific Conclusion: 
        This Planet is unlikely to sustain stable surface liquid water or Earth-like biosignatures under current known conditions.
        ''' 
    

    if hab_pred==1:
        hab_class='HABITABLE'

        interpretation=f'''
        Planet Classification: {hab_class}
        
        Confidence Level: {confidence:.2%}

        Inerpretation:
        This Exo-Planet demonstrates promising Earth-like habitability characteristics based on learned astrophysical patterns.

        Positive indicators may includes:
        - Suitable equilibrium temperature
        - Favourable stellar flux
        - Potential habitable zone placement
        - Stable orbital characteristics
        - Earth-like planetary radius or mass
        - Supportive stellar conditions

        Scientific Conclusion:
        This planet may be a strong candidate for further astrobiological study, atmospheric analysis, and future obervational prioritization.
    '''

    return interpretation


def predict_esi(X):
    esi_pred=esi_model.predict(X)[0]

    if esi_pred>=0.80:
        category= "Highly Earth-like"
        interpretation='''
        Interpretation:
        This Planet shows very strong similarity to Earth and may posses highly favourable conditions for habitability.'''

    elif esi_pred>=0.60:
        category="Moderately Earth-Like"
        interpretation='''
        Interpretation:
        This Planet shares several Earth-like characteristics but may have environmental or structural limitations.'''
    
    elif esi_pred>=0.40:
        category='Low Earth Similarity'
        interpretation='''
        Interpretation: 
        This planet has some potentially favorable features, but overall habitability conditions may be limited.
        '''

    else:
        category='Hostile / Non-Earth-Like'
        interpretation='''
        Interpretation: 
        This planet is significantly different from Earth and is unlikely to support Earth-like Life.
        '''

    return f'''
    EARTH SIMILARITY INDEX (ESI) is {esi_pred:.3f}

    Habitability Category : {category}

    {interpretation}

    Scientific Meaning: 
    ESI is a comparative metric that estimates how physically similar an ExoPlanets is to Earth using radius, density, escape velocity, and surface temperature indicators.
    '''


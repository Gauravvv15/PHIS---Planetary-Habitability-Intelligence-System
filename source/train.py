import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score, mean_squared_error
from preprocessing_df import preprocessing_train

nasa_df=pd.read_csv('data/nasa_df.csv')
phl_df=pd.read_csv('data/phl_exoplanet.csv')

df=preprocessing_train(nasa_df, phl_df)



# separating x and y
X=df.drop(['habitable','esi'], axis=1)
Y=df[['habitable','esi']].copy()

x_train, x_test, y_train, y_test=train_test_split(
    X, Y, 
    test_size=0.2,
    random_state=42,
    stratify=Y['habitable']
)

y_train_class=y_train['habitable']
y_test_class=y_test['habitable']

y_train_reg=y_train['esi']
y_test_reg=y_test['esi']

df_cols=X.columns.tolist()


# next step is scaling the features

scaler=RobustScaler()
x_train_scaled=pd.DataFrame(
    scaler.fit_transform(x_train),
    columns=x_train.columns,
    index=x_train.index
)
 
#for x_test
x_test_scaled=pd.DataFrame(
    scaler.transform(x_test),
    columns=x_test.columns,
    index=x_test.index
)


def train_classification_model(x_train_scaled, y_train_class, x_test_scaled, y_test_class):
    #using smote for data imbalance
    smote=SMOTE(random_state=42, k_neighbors=5)
    x_train_bal, y_train_bal=smote.fit_resample(x_train_scaled, y_train_class)

    y_train_bal.value_counts()

    #xgb classifier

    xgb_class=XGBClassifier(
        n_estimators=600,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.8,
        reg_lambda= 5.0,
        reg_alpha=0,
        min_child_weight=1,
        gamma=0.2,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train_bal[y_train_bal==0]) / len(y_train_bal[y_train_bal==1]),
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    xgb_class.fit(x_train_bal, y_train_bal)

    xgb_pr=xgb_class.predict(x_test_scaled)
    xgb_acc=accuracy_score(y_test_class, xgb_pr)
    print(f'accuracy score of XGBclassifier:  {xgb_acc}')
    print(classification_report(y_test_class,xgb_pr))

    joblib.dump(xgb_class, 'models/xgb_habitability_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(df_cols, 'models/df_cols.pkl')
   

def train_regression_model(x_train_scaled, y_train_reg, x_test_scaled, y_test_reg):
    xgb_reg=XGBRegressor(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.1,
        gamma=0,
        colsample_bytree=0.9,
        subsample=0.8,
        reg_lambda=5.0,
        reg_alpha=0,
        random_state=42,
        n_jobs=-1
    )
    xgb_reg.fit(x_train_scaled, y_train_reg)

    reg_pred=xgb_reg.predict(x_test_scaled)
    mae=mean_absolute_error(y_test_reg, reg_pred)
    mse=mean_squared_error(y_test_reg, reg_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test_reg, reg_pred)

    print(f'mean_absolute_error: {mae}')
    print(f'mean_squared_error: {mse}')
    print(f'rmse: {rmse}')
    print(f'r2 score: {r2}')

    joblib.dump(xgb_reg, 'models/xgb_esi_model.pkl')


train_classification_model(x_train_scaled, y_train_class, x_test_scaled, y_test_class)
train_regression_model(x_train_scaled, y_train_reg, x_test_scaled, y_test_reg)
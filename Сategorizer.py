
from xgboost import XGBClassifier
from numpy import transpose, unique, argmax, argmin, append, gradient, where, diff, sign, delete, ndarray, unique

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from statistics import median

from scipy.stats import chi2_contingency
from scipy.ndimage import gaussian_filter1d

from shap import TreeExplainer, KernelExplainer

import pandas as pd
from copy import copy

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import StackingClassifier

from pickle import load, dump
from os.path import exists


class Categorizer:  #Класс категоризатора. На основаннии набора данных получает точки интереса.

    model = XGBClassifier()

    use_auc = True
    use_p_value = True
    use_centroid = True
    use_shap = True
    smooth = 5
    d_cross = 0
    file_model = ""

    def __init__(self,

                 use_auc = True, #Использовать пароги Max(AUC)
                 use_p_value = True, #Использовать пароги Min(p_value)
                 use_centroid = True, #Использовать пароги центроиды
                 use_shap = True, #Использовать пароги SHAP

                 smooth = 5, #Степень сглаживания SHAP графика.

                 #Гиперпараметры модели classxgboost.XGBClassifier более подробно описано тут: https://xgboost.readthedocs.io/en/stable/python/python_api.html
                 eval_metric = "aucpr", learning_rate = None, scale_pos_weight = None, max_depth = None,
                 n_estimators = None, random_state = None, verbosity = None, objective = 'binary:logistic', booster = 'gbtree',
                 tree_method = "exact", max_delta_step = None, gamma = None,
                 min_child_weight = None, subsample = None, colsample_bylevel = None, d_cross = 0, file_model = ""
                 ):

      self.use_auc = use_auc
      self.use_p_value = use_p_value
      self.use_centroid = use_centroid
      self.use_shap = use_shap
      self.smooth = smooth
      self.d_cross = d_cross

      self.file_model = file_model

      self.model = XGBClassifier(       eval_metric = eval_metric, learning_rate=learning_rate,
                                        scale_pos_weight = scale_pos_weight, max_depth=max_depth,  n_estimators=n_estimators,
                                        random_state=random_state,
                                        verbosity=verbosity , objective= objective, booster= booster,
                                        tree_method = tree_method, max_delta_step=max_delta_step, gamma=gamma, min_child_weight=min_child_weight,
                                        subsample=subsample, colsample_bylevel=colsample_bylevel,
                                        #colsample_by_level = 0.75,
                                        )     

    #Получить точки интереса на основании набора данных
    # x_data - массив из строк таблицы (массивов) описывающих параметры объекты
    # y_data - массив из строк таблицы (массивов) описывающих ответ решаемой задачи объекты
    def getPointsOfInterest( self, x_data, y_data, grope_rule = True ):

      points_of_interest = []
      if( self.use_auc or self.use_p_value ):
        points_of_interest_using_optimization = self.getPointsOfInterestUsingOptimization( x_data, y_data )
        points_of_interest = points_of_interest_using_optimization

      if( self.use_centroid ):
        points_of_interest_using_centroid = self.getPointsOfInterestUsingCentroid( x_data, y_data )
        points_of_interest = append( points_of_interest, points_of_interest_using_centroid )

      if( self.use_shap ):
        points_of_interest_using_shap = self.getPointsOfInterestUsingSHAP( x_data, y_data )
        points_of_interest = append( points_of_interest, points_of_interest_using_shap )

      if grope_rule:

        table_risk_factor = {}

        for predictor in range(0, len(x_data[ 0 ])):
          table_risk_factor[predictor] = []

        for point_of_interest in points_of_interest:
          table_risk_factor[ point_of_interest["pred"]].append( point_of_interest )

        return table_risk_factor

      else:
        return points_of_interest

    def getPointsOfInterestUsingOptimization( self, x_data, y_data ):

        x_data = transpose(x_data)
        result = []
        for predictor_index, predictor in enumerate(x_data):
          unique_values = unique(predictor)

          dichotomization_data = self.getDichotomizationData( predictor, unique_values )
          if( self.use_auc ):
            auc = self.getAUC( dichotomization_data, y_data )
            max_auc_index = argmax(auc)
            result.append( { "value": unique_values[max_auc_index], "pred": predictor_index  } )

          if( self.use_p_value ):
            p_value = self.getPValue( dichotomization_data, y_data )
            min_p_value_index = argmin(p_value)
            result.append( { "value": unique_values[min_p_value_index], "pred": predictor_index  } )

        return result

    def getPointsOfInterestUsingCentroid( self, x_data, y_data ):

          x_data = transpose(x_data)
          result = []
          dataset = pd.DataFrame()
          y_data = transpose(y_data)[0]
          dataset["y"] = y_data

          for predictor_index, predictor in enumerate(x_data):

            dataset[predictor_index] = predictor

            group_1 = dataset.loc[(dataset["y"] == 1)][predictor_index]
            group_0 = dataset.loc[(dataset["y"] == 0)][predictor_index]

            if ( len(group_0) == 0 or len(group_1) == 0 ):
              continue

            group_1_median = median( group_1 )
            group_0_median = median( group_0 )

            result.append( { "value": (group_1_median+group_0_median)/2, "pred": predictor_index  } )

          return result

    def getPointsOfInterestUsingSHAP( self, x_data, y_data ):

        result = []
        #d = "D:\Work\Science\Dissertation\Dissertation\pickles\\" + self.file_model
        #if exists(d):
        #    try: 
        #        with open(d, 'rb') as f:
        #            explainer = load(f)
        #    except Exception as e:
        #        print(e, "explainer not load")
        #else:
        self.model.fit( x_data, y_data)
        explainer = TreeExplainer( self.model )

        #    with open(d, 'wb') as f:
        #        dump( explainer, f )
        
        shap_values = explainer.shap_values( x_data )
        shap_values = transpose(shap_values)
        x_data = transpose(x_data)

        for feature in range( 0, len(shap_values) ):

          dataset = pd.DataFrame( )
          dataset["p"] = x_data[feature]
          dataset["shap"] = shap_values[feature]

          dataset = dataset.groupby(['p']).mean()
          dataset = dataset.sort_index()

          dataset["shap"] = gaussian_filter1d(dataset["shap"], self.smooth[feature] )

          dataset["smooth_d2"] = gradient(gradient(dataset["shap"]))


          points = where(diff(sign(dataset["smooth_d2"])))[0]
          infls = []
          for point in points:
              infls.append(dataset.index[point])

          old_index = None          
          shap_max = max(dataset["shap"])
          dataset = dataset.loc[ dataset["shap"] > 0 ]
          cross_point = []
          for index in dataset.index[1:-1]:

            if old_index == None:
              old_index = index
              continue
            
            shap_value = dataset["shap"][index]
            old_shap_value = dataset["shap"][old_index]


            value = (old_index+index)/2
            cross = 0
            while cross <= shap_max:
              if ((shap_value >= cross and old_shap_value <= cross) or (old_shap_value >= cross and shap_value <= cross )):
                cross_point.append( value )
              cross = cross + self.d_cross[feature]

            old_index = index

          infls = append( infls, unique(cross_point)  )          
          
          if len( infls ) > 0:
            min_infls = min(infls)
            max_infls = max(infls)
            infls.sort()
            infls_res = copy(infls)
            infl_old = None
            for infl in infls:
                if( infl_old != None and abs( infl_old - infl )/( max_infls - min_infls )<=0.005 ):
                    infls_res = delete( infls_res, where( infls_res == infl))           
                else:
                    infl_old = infl
            infls = infls_res

          
          
          for infl in infls:
            if( infl != min(x_data[feature]) and infl != max(x_data[feature])  ):
              result.append( {'value': infl, 'pred':feature } )

        return result

    def getDichotomizationData( self, predictor, unique_values ):
        dichotomization_data = []
        for unique_value in unique_values:
          dichotomization_data.append( predictor > unique_value )
        return dichotomization_data

    def getAUC( self, dichotomization_data, y_data ):
        result = []
        skf = StratifiedKFold(n_splits=5)

        for x_data in dichotomization_data:
          auc_array = []
          for train,test in skf.split( x_data, y_data ):
            y_train = y_data[train]
            x_train = x_data[train].reshape(-1, 1)
            y_test = y_data[test]
            x_test = x_data[test].reshape(-1, 1)

            logistReg = LogisticRegression()
            logistReg.fit( x_train, y_train.ravel() )
            y_test_pred = logistReg.predict_proba(x_test)[:,1]
            auc_array.append( roc_auc_score( y_test, y_test_pred ) )

          result.append( median(auc_array) )

        return result

    def getPValue( self, dichotomization_data, y_data ):
        result = []
        y_data = transpose(y_data)[0]

        for x_data in dichotomization_data:

          data_crosstab = pd.crosstab( y_data,
                                       x_data)
          if( data_crosstab.shape == (2, 2) ):
            chi2, p_value, df, expected = chi2_contingency(data_crosstab)
          else:
            p_value = 1

          result.append( p_value )

        return result


#Usage example
#Selecting a data set
#df = dataset[["Age","ЧСС (b)","Смерть Рез"]]
#df = df.dropna()
#x_data = df[["Age","ЧСС (b)"]].values
#y_data = df[["Смерть Рез"]].values
#Launch of the categorizator
#categorizer = Сategorizer()
#points = categorizer.getPointsOfInterest( x_data, y_data )
#print(points)

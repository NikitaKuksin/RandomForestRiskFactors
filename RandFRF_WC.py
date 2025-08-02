
from numpy import transpose, mean
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


from RandFRF.ModifyTree import TreeClassifier, getGain
from RandFRF.Сategorizer import Categorizer

#Класс модифицированного "Случайного леса". Модель
class RandomFRF_WC:

    trees = None
    categorizer = None
    fit_dataset_x = None
    fit_dataset_y = None
    fit_dataset = None
    params = None

    min_samples_leaf_tree = None
    class_values = None
    use_predictor_once = None
    getMetric = getGain
    use_auc = True
    use_p_value = True
    use_centroid = True
    use_shap = True
    smooth = 5
    d_cross = 0

    eval_metric = "auc"
    learning_rate = None
    scale_pos_weight = None
    max_depth = None
    n_estimators = None
    random_state = None
    verbosity = None
    objective = 'binary:logistic'
    booster = 'gbtree'
    tree_method = "exact"
    max_delta_step = None
    gamma = None
    min_child_weight = None
    subsample = None
    colsample_bylevel = None

    predictor_rules = None
    coef = None
    pred_count = None
    drop_columns = None

    sum_max_coef = 0

    cutoff_risc_factor = 0
    fit_dataset_red = None
    
    file_model = ""

    def __init__(self,    
                          
                          smooth, #Степень сглаживания SHAP графика. Для каждого предиктора задается отдельно.
                          d_cross, #Шаг разделения SHAP графика. Для каждого предиктора задается отдельно. Для каждого предиктора задается отдельно.
                          cutoff_risc_factor, #Порог отсечения фактра риска. Для каждого предиктора задается отдельно. 

                          min_samples_leaf_tree, #Минимальное количество объектов необходимое для создания узла (листа) дерева. Для каждого предиктора задается отдельно.

                          class_values = None, #Класс по которому расчитывается метрика качества разделения (0, 1). Если указано None учитываются оба класса
                          use_predictor_once = True, #Использовать предиктор в ветви только один раз
                          getMetric = getGain, #Ссылка на функцию расчета метрики разделения

                          use_auc = True, #Использовать пароги Max(AUC)
                          use_p_value = True, #Использовать пароги Min(p_value)
                          use_centroid = True, #Использовать пароги центроиды
                          use_shap = True, #Использовать пароги SHAP
                          
                          #Гиперпараметры модели classxgboost.XGBClassifier более подробно описано тут: https://xgboost.readthedocs.io/en/stable/python/python_api.html
                          eval_metric = "auc", learning_rate = None, scale_pos_weight = None, max_depth = None, n_estimators = None,
                          random_state = None, verbosity = None, objective = 'binary:logistic', booster = 'gbtree', tree_method = "exact", max_delta_step = None, gamma = None, min_child_weight = None,
                          subsample = None, colsample_bylevel = None, 
                          
                         ):

      self.min_samples_leaf_tree = min_samples_leaf_tree
      self.class_values = class_values
      self.use_predictor_once = use_predictor_once
      self.getMetric = getMetric
      self.use_auc = use_auc
      self.use_p_value = use_p_value
      self.use_centroid = use_centroid
      self.use_shap = use_shap
      self.smooth = smooth
      self.d_cross = d_cross

      self.eval_metric = eval_metric
      self.learning_rate = learning_rate
      self.scale_pos_weight = scale_pos_weight
      self.max_depth = max_depth
      self.n_estimators = n_estimators
      self.random_state = random_state
      self.verbosity = verbosity
      self.objective = objective
      self.booster = booster
      self.tree_method = tree_method
      self.max_delta_step = max_delta_step
      self.gamma = gamma
      self.min_child_weight = min_child_weight
      self.subsample = subsample
      self.colsample_bylevel = colsample_bylevel
      self.cutoff_risc_factor = cutoff_risc_factor

    #Обучение модели леса на основании набора данных.
      # x_data - массив из строк таблицы (массивов) описывающих параметры объекты
      # y_data - массив из строк таблицы (массивов) описывающих ответ решаемой задачи объекты
    def fit( self, x_data, y_data ):

      self.fit_dataset_x = x_data
      self.fit_dataset = pd.DataFrame(x_data)


      self.fit_dataset_y = y_data
      self.fit_dataset["y"] = y_data
      self.trees = []

      categorizer = Categorizer(      use_auc = self.use_auc, use_p_value = self.use_p_value, use_centroid = self.use_centroid, use_shap = self.use_shap, smooth = self.smooth,
                                        eval_metric = self.eval_metric , learning_rate = self.learning_rate, scale_pos_weight = self.scale_pos_weight, max_depth = self.max_depth,
                                        n_estimators = self.n_estimators, random_state = self.random_state, verbosity = self.verbosity, objective = self.objective, booster = self.booster,
                                        tree_method = self.tree_method, max_delta_step = self.max_delta_step, gamma = self.gamma,
                                        min_child_weight = self.min_child_weight, subsample = self.subsample, colsample_bylevel = self.colsample_bylevel, d_cross = self.d_cross, file_model = self.file_model )
      rules = categorizer.getPointsOfInterest( x_data, y_data, grope_rule = True )

      semples = self.getSemples( x_data )

      for column in range(0,len(x_data[0])):
        tree = TreeClassifier( min_samples_leaf_tree=self.min_samples_leaf_tree[column], class_value = self.class_values[column] )
        tree.fit( semples[column], y_data, rules = rules[column], columns_name = [ column ] )
        self.trees.append({ "model":tree })
        
      #----------------------------------------------------------------------------------------------Обучение логистической регресии и выделение факторов риска
      self.predictor_rules = self.getRules( )

      df_train = pd.DataFrame( x_data )
      self.pred_count = len(x_data[0])

      max_pred_prob = {}
      self.sum_max_prob = 0


      df_train["result"] = y_data.ravel()
      df_train = df_train.dropna()
      y_data = df_train["result"].values
      df_train = df_train.drop('result', axis=1)

      for predict in range(0,self.pred_count):

        max_pred_prob[predict] = 0

        for rule in self.predictor_rules[predict]:


          list_rule = self.predictor_rules[predict][rule]["List_of_RF"]
          min_val = 0
          df_train[str(predict)+str(rule)] = min_val
          for r in list_rule:
            min_val = min_val - 1
            if r["sign"] == "<":
              df_train.loc[ ( (df_train[predict] < r["value"]) & (df_train[str(predict)+str(rule)] == min_val+1 ) ), (str(predict)+str(rule)) ] = min_val
            else:
              df_train.loc[ ( (df_train[predict] >= r["value"]) & (df_train[str(predict)+str(rule)] == min_val+1 ) ), (str(predict)+str(rule)) ] = min_val

          df_train.loc[ ( df_train[str(predict)+str(rule)] != min_val ), (str(predict)+str(rule)) ] = 0.0
          df_train[str(predict)+str(rule)] = df_train[str(predict)+str(rule)].astype(float)
          df_train.loc[ ( df_train[str(predict)+str(rule)] == min_val ), (str(predict)+str(rule)) ] = self.predictor_rules[predict][rule]["Probability"][1]


      logistReg = LogisticRegression()
      df_train_fist = df_train.drop( columns=df_train.columns[:self.pred_count] ).values

      logistReg.fit( df_train_fist, y_data )
      self.coef = logistReg.coef_[0]
      self.drop_columns = []
      self.df_train = df_train

      i = 0
      max_pred_coef = {}
      self.sum_max_coef = 0
      for predict in range(0,self.pred_count):
        max_pred_coef[predict] = 0
        for rule in self.predictor_rules[predict]:

          self.predictor_rules[predict][rule]["Coef"] = self.coef[ i ]
          if self.coef[ i ] > max_pred_coef[predict]:
            max_pred_coef[predict] = self.coef[ i ]
          i = i + 1
        self.sum_max_coef = self.sum_max_coef + max_pred_coef[predict]

      for i in range( 0, self.pred_count+1 ):
          self.drop_columns.append( df_train.columns[  i ] )


      for i in range( 0, len(self.coef) ):

        if self.coef[ i ] <= self.cutoff_risc_factor:
          self.drop_columns.append( df_train.columns[ self.pred_count+i ] )

      self.x_train = df_train.drop( columns=self.drop_columns ).values
      
      self.line_model = LogisticRegression()
      self.line_model.fit( self.x_train, y_data )

      max_pred_coef = {}
      self.sum_max_coef = 0
      self.sum_min_coef = 0
      self.coef = self.line_model.coef_[0]
      i = 0
      for predict in range(0,self.pred_count):
        max_pred_coef[predict] = 0
        for rule in self.predictor_rules[predict]:

           if self.predictor_rules[predict][rule]["Coef"] > 0:

             if( self.coef[i] > 0 ):
                self.predictor_rules[predict][rule]["Coef"] = self.coef[i]
             else:
                self.predictor_rules[predict][rule]["Coef"] = 0
                  

             if self.coef[i] > max_pred_coef[predict]:
               max_pred_coef[predict] = self.predictor_rules[predict][rule]["Coef"]
             
             i = i + 1

        self.sum_max_coef = self.sum_max_coef + max_pred_coef[predict]
      #----------------------------------------------------------------------------------------------

    def getSemples( self, x_data ):
      x_data = transpose(x_data)
      semples = {}
      for column in range(len(x_data)):
        semples[column] = x_data[column]
      return semples

    def getRulesSet( self, dataset, rules, used_rules ):

      rule = None
      e = self.getEntropy( dataset, "y" )

      for r in rules:
        dataset_test = copy(dataset)

        if r[0] != None:
          dataset_test = dataset_test.loc[ dataset_test[ r[0]["pred"] ] >= r[0]["value"]  ]

        if r[1] != None:
          dataset_test = dataset_test.loc[ dataset_test[ r[1]["pred"] ] <= r[1]["value"]  ]

        if not len( dataset_test.values )>15:
          continue

        e_n = self.getEntropy( dataset_test, "y" )
        if e_n > e:
          e = e_n
          rule = r
          ds = dataset_test
          if( e == 1 ):
            break

      if( rule != None ):
        rules.remove( rule )
        used_rules.append( rule )
        res = self.getRulesSet( ds, rules, used_rules )
        dataset = res["dataset"]
        used_rules = res["used_rules"]

      return {"dataset": dataset, "used_rules":used_rules}

    def predict_proba( self, x_test ):

      df_test = pd.DataFrame( x_test )
      min_max_for_predict = {  }

      for predict in range(0,self.pred_count):
        for rule in self.predictor_rules[predict]:
          list_rule = self.predictor_rules[predict][rule]["List_of_RF"]
          min_val = 0
          df_test[str(predict)+str(rule)] = min_val
          for r in list_rule:
            min_val = min_val - 1
            if r["sign"] == "<":
              df_test.loc[ ( (df_test[predict] < r["value"]) & (df_test[str(predict)+str(rule)] == min_val+1 ) ), (str(predict)+str(rule)) ] = min_val
            else:
              df_test.loc[ ( (df_test[predict] >= r["value"]) & (df_test[str(predict)+str(rule)] == min_val+1 ) ), (str(predict)+str(rule)) ] = min_val

          df_test.loc[ ( df_test[str(predict)+str(rule)] != min_val ), (str(predict)+str(rule)) ] = 0
          df_test[str(predict)+str(rule)] = df_test[str(predict)+str(rule)].astype(float) 

          df_test.loc[ ( df_test[str(predict)+str(rule)] == min_val ), (str(predict)+str(rule)) ] = self.predictor_rules[predict][rule]["Coef"] / self.sum_max_coef

      x_test = df_test.drop( columns=self.drop_columns ).values

      y_test_pred = self.get_prob_coef(x_test)
      
      return y_test_pred

    def get_p( self, p ):

      if len( p ) > 1:
        res = p[0] + self.get_p( p[1:] )
      else:
        res = p[0]

      if len( p ) > 1:
        res2 = p[0] * self.get_p( p[1:] )
      else:
        res2 = 0
      return res - res2

    def get_prob_coef( self, x_test):
      result = []
      for x_data in x_test:
        if len(x_data)>0:
          result.append( [ 1-mean(x_data), mean(x_data) ] )
        else:
          result.append( [ 1, 0 ] )

      return result

    def getRules( self ):

      rules = {}
      x = []
      for i in range( 0, len( self.trees ) ):
          x.append(i)
          rules[i] = self.trees[i]["model"].get_rules( x )

      return rules


#      Usage example
#      dataset = pd.read_excel (path)         
#      dataset = dataset[["X1","X2","X3","Y"]]
#      dataset = dataset.dropna()
#      x_data = dataset[["X1","X2","X3"]].values
#      y_data = dataset[["Y"]].values

#      model = RandomFRF_WC(                     random_state=42, eval_metric="auc",
#                                                max_delta_step=3,
#                                                n_estimators = 225,
#                                                learning_rate = 0.025,
#                                                max_depth = 5,
#                                                subsample = 0.025,
#                                                min_samples_leaf_tree = [30, 45, 60],
#                                                use_auc=True,
#                                                use_centroid=True,
#                                                use_p_value=True,
#                                                smooth = [3, 5, 7],
#                                                cutoff_risc_factor = [0.1, 0.1, 0.1],
#                                                d_cross = [ 0, 0, 0 ] ,
#                                                class_values = [0, 0, 0],
#                                  )
#
#    model.fit( x_data, y_data )
#
#    proba = model.predict_proba( x_data )
#    proba = pd.DataFrame( proba )
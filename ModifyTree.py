



from numpy import delete, where, ndarray
from copy import deepcopy
from math import log
from copy import copy
import pandas as pd
from RandFRF.Сategorizer import Categorizer

#Класс листа дерева
class TreeLeaf:

    left_rule = None
    right_rule = None

    def __init__(self):
      self.probabilities = { 0: 0, 1: 0 }

    def fit( self, dataset, y, left_rule, right_rule ):

      self.left_rule = left_rule
      self.right_rule = right_rule

      values = dataset[y].unique()

      self.probabilities[0] = len(dataset.loc[dataset[y]==0])/len(dataset)
      self.probabilities[1] = len(dataset.loc[dataset[y]==1])/len(dataset)

    def predict( self, predictors ):
      #return {0: self.probabilities[0], 1: self.probabilities[1], "rules":[ self.left_rule, self.right_rule ] }
      return [ self.probabilities[0], self.probabilities[1] ]

    def print( self, ax, point, parametrs ):

      width = parametrs["width"]
      height = parametrs["height"]
      bord = parametrs["bord"]
      x_lim = parametrs["x_lim"][1]
      nodes_on_level = parametrs["nodes_on_level"]

      color = (1,1,1)
      if 1 in self.probabilities.keys():
        f_1 = self.probabilities[1]
        color = (1,1-f_1,1-f_1)

      rect = Rectangle((point["x"]-width/2, point["y"]-height), width, height, edgecolor = "black" , facecolor = color)

      ax.add_patch(rect)

      rx, ry = rect.get_xy()
      cx = rx + rect.get_width()/2.0
      cy = ry + rect.get_height()/2.0

      keys = list(self.probabilities.keys())
      keys.sort()
      text = ""

      for i in keys:
        text =  text + "{0}: {1:.3f}\n".format(i,self.probabilities[i])
      text = text[0:-1]

      ax.annotate( text , (cx, cy), color='black', weight='bold', fontsize=parametrs["fs"], ha='center', va='center')

    def get_max_depth( self, level = 0 ):
      level = level + 1
      return level

    def get_rules( self, global_rules, rul_n = 1, x = None ):
      if not "Rule_"+str(rul_n) in global_rules.keys():
        return None
      global_rules["Rule_"+str(rul_n)]["Probability"] = self.probabilities

      #Метрика Прирост информаии

#Metric Gain
def getGain( datasets, y, class_value ):
          #print(class_value)
          entropy_left = getEntropy( datasets["left"], y, class_value )
          entropy_right = getEntropy( datasets["right"], y, class_value )
          entropy_full = getEntropy( datasets["full"], y, class_value )

          len_dataset = len(datasets["left"]) + len(datasets["right"])

          entropy_mean = ( len(datasets["left"]) / len_dataset * entropy_left + entropy_right * len(datasets["right"]) / len_dataset) / 2
          gain = entropy_full - entropy_mean

          return gain

def getEntropy( dataset, y, class_value ):

      if( class_value == None ):
        dataset_y = dataset[y]
        sum = 0
        count_y = len(dataset_y)
        for value in dataset_y.unique():
          count_value = len(dataset_y.loc[ dataset[y] == value ])
          frequency = count_value / count_y
          log_frequency = log(frequency)
          sum = sum+frequency*log_frequency
        sum = -sum
        return sum
      else:
        count_y = len(dataset)
        count_value = len(dataset.loc[ dataset[y] == class_value ])
        frequency = count_value / count_y
        if(frequency!=0):
          log_frequency = log(frequency)
        else:
          log_frequency = 0
        return -frequency*log_frequency

#Metric Gini Index
def getGini( dataset, y, class_value ):

      if( class_value == None ):
        sum = 0
        count_y = len(dataset)
        for value in dataset[y].unique():
          count_value = len(dataset.loc[ dataset[y] == value ])
          frequency = count_value / count_y
          sum = sum+frequency*(1-frequency)
        return -sum
      else:
        count_y = len(dataset)
        count_value = len(dataset.loc[ dataset[y] == class_value ])
        frequency = count_value / count_y
        return -frequency*(1-frequency)

#Класс узла дерева
class TreeNode:

    left_child = None
    right_child = None
    rule = None
    probabilities = None
    level = 0
    nomber_on_level = 0
    volume = 0
    probabilities = 0
    min_samples_leaf = 0

    def __init__(self, min_samples_leaf = 0):
        self.left_child = None
        self.right_child = None
        self.rule = None
        self.probabilities = None
        self.level = 0
        self.nomber_on_level = 0
        self.volume = 0
        self.probabilities = 0
        self.min_samples_leaf = min_samples_leaf

    def getRightLeftDataset( self, dataset, pred , rule_value ):

      #self.min_samples_leaf = 131
      #print(self.min_samples_leaf)
      dataset_left_rules = dataset.loc[ dataset[ pred ] < rule_value ]
      if( len(dataset_left_rules) == 0 or len(dataset_left_rules) < self.min_samples_leaf ):
        return None

      dataset_right_rules = dataset.loc[ dataset[ pred ] >= rule_value ]
      if( len(dataset_right_rules) == 0 or len(dataset_right_rules) < self.min_samples_leaf ):
        return None

      return { "left":dataset_left_rules, "right":dataset_right_rules, "full": dataset }
    
    def fit( self, dataset, y, rules, getMetric = getGain, level = 0, use_predictor_once = False, parent_rule = [], class_value = None, left_rule = None, right_rule = None, min_samples_leaf = 50 ):

      optimal_value = 0
      for rule in rules:

          if( use_predictor_once and rule["pred"] in parent_rule ):
              continue
          datasets = self.getRightLeftDataset( dataset, rule["pred"] , rule["value"] )

          if datasets == None :
            rules = delete( rules, where( rules == rule))
            continue

          metric_value = getGain( datasets, y, class_value )

          if optimal_value < metric_value:

            self.rule = rule
            optimal_value = metric_value
            dataset_left = datasets["left"]
            dataset_right = datasets["right"]

      if( self.rule == None ):
        return False

      rules = delete( rules, where( rules == self.rule))

      if( len(rules) != 0 ):
        self.left_child = TreeNode( self.min_samples_leaf )
        fit_success = self.left_child.fit( copy(dataset_left), y, copy(rules), getMetric = getMetric, level = level+1, use_predictor_once = use_predictor_once,
                                           parent_rule = parent_rule, class_value = class_value, left_rule = left_rule,
                                           right_rule = right_rule )
        if ( not fit_success ):
          self.left_child = None


      if( self.left_child == None ):
        self.left_child = TreeLeaf()
        self.left_child.fit( copy(dataset_left), y, left_rule, right_rule )

      if( len(rules) != 0 ):
        self.right_child = TreeNode( self.min_samples_leaf )
        fit_success = self.right_child.fit( copy(dataset_right), y, copy(rules), getMetric = getMetric, level = level+1, use_predictor_once = use_predictor_once,
                                            parent_rule = parent_rule, class_value = class_value, left_rule = left_rule,
                                            right_rule = right_rule )
        if ( not fit_success ):
          self.right_child = None

      if( self.right_child == None ):
        self.right_child = TreeLeaf()
        self.right_child.fit( copy(dataset_right), y, left_rule, right_rule )
      return True

    def get_max_depth( self, level = 0 ):

      level = level + 1

      left_depth = self.left_child.get_max_depth( level )
      right_depth = self.right_child.get_max_depth( level )

      if left_depth >= right_depth:
        max_depth = left_depth
      else:
        max_depth = right_depth
      return max_depth

    def get_volume( self ):
      self.volume = self.left_child.get_volume( )
      self.volume = self.volume + self.right_child.get_volume( )
      self.volume = self.volume + 1
      return self.volume

    def get_n_leaves( self ):
      res = 0
      res = self.left_child.get_n_leaves( )
      res = res +  self.right_child.get_n_leaves( )
      return res

    def get_nodes_on_level( self, nodes_on_level ):

      nodes_on_level[self.level] = nodes_on_level[self.level] + 1
      self.nomber_on_level = nodes_on_level[self.level]

      self.left_child.get_nodes_on_level( nodes_on_level )
      self.right_child.get_nodes_on_level( nodes_on_level )

      return None

    def predict( self, predictors ):

      predictor = predictors[ self.rule[ "pred" ] ]

      if( predictor < self.rule[ "value" ] ):
        probabilities = self.left_child.predict( predictors )
      else:
        probabilities = self.right_child.predict( predictors )

      return probabilities

    def get_rules( self, global_rules, rul_n = 1, x = None):

      if not "Rule_"+str(rul_n) in global_rules.keys():
        global_rules["Rule_"+str(rul_n)] = { "List_of_RF":[], "Probability":{}, "AUC":{}, "Weight":{} }

      rules = copy( global_rules["Rule_"+str(rul_n)][ "List_of_RF" ] )

      rule_left = copy(self.rule)

      if x != None:
        rule_left['pred'] = x[rule_left['pred']]

      rule_left["sign"] = "<"

      global_rules["Rule_"+str(rul_n)][ "List_of_RF" ].append( rule_left )

      self.left_child.get_rules( global_rules, rul_n, x )

      last_rule = len(global_rules.keys())

      global_rules["Rule_"+str(last_rule+1)] = { "List_of_RF":rules, "probabilities":{}, "AUC":{}, "Weight":{} }

      rule_right = copy(self.rule)
      rule_right["sign"] = ">="
      if x != None:
        rule_right['pred'] = x[rule_right['pred']]

      global_rules["Rule_"+str(last_rule+1)][ "List_of_RF" ].append( rule_right )

      self.right_child.get_rules( global_rules, last_rule+1, x )

    def print( self, ax, point, parametrs ):

      width = parametrs["width"]
      height = parametrs["height"]
      predict = parametrs["predict"]
      bord = parametrs["bord"]
      nodes_on_level = parametrs["nodes_on_level"]
      x_lim = parametrs["x_lim"]

      rect = Rectangle((point["x"]-width/2, point["y"]-height), width, height, fill=None )

      ax.add_patch(rect)

      rx, ry = rect.get_xy()
      cx = rx + rect.get_width()/2.0
      cy = ry + rect.get_height()/2.0
      ax.annotate( str(predict[self.rule["pred"]]) + ">=" + str(self.rule["value"]) , (cx, cy), color='black', weight='bold', fontsize=parametrs["fs"], ha='center', va='center')
      new_parametrs = deepcopy(parametrs)
      ax.annotate( "{0}: {1:.3f}\n".format(1,self.probabilities[1]) ,  (cx, cy+rect.get_height()/2.0), color='black', weight='bold', fontsize=new_parametrs["fs"], ha='center', va='center')

      if self.left_child != None:

        otnosenie_l=self.left_child.volume/(self.left_child.volume+self.right_child.volume)
        new_parametrs["x_lim"] = [x_lim[0], x_lim[0] + (x_lim[1] - x_lim[0]) * otnosenie_l ]

        x_line = [ point["x"] , (new_parametrs["x_lim"][0]+new_parametrs["x_lim"][1])/2-width/2 ]
        y_line = [ point["y"]-height , point["y"]-height*2 ]
        ax.plot( x_line, y_line, color="gray")
        ax.annotate( "False" , ((x_line[0]+x_line[1])/2, (y_line[0]+y_line[1])/2), color='black', weight='bold', fontsize=new_parametrs["fs"], ha='center', va='center')
        self.left_child.print( ax, { "x":(new_parametrs["x_lim"][0]+new_parametrs["x_lim"][1])/2-width/2,"y":point["y"]-height*2 }, deepcopy(new_parametrs) )

      new_parametrs = deepcopy(parametrs)
      if self.right_child != None:

        new_parametrs["x_lim"] = [x_lim[0] + (x_lim[1] - x_lim[0]) * otnosenie_l, x_lim[1] ]

        x_line = [ point["x"] ,  (new_parametrs["x_lim"][0]+new_parametrs["x_lim"][1])/2-width/2 ]
        y_line = [ point["y"]-height , point["y"]-height*2 ]
        ax.plot( x_line, y_line, color="gray")
        ax.annotate( "True" , ((x_line[0]+x_line[1])/2, (y_line[0]+y_line[1])/2), color='black', weight='bold', fontsize=new_parametrs["fs"], ha='center', va='center')
        new_parametrs["x_lim"] = [ x_lim[0] + (x_lim[1] - x_lim[0]) * otnosenie_l  , x_lim[1]]
        self.right_child.print( ax, { "x":(new_parametrs["x_lim"][0]+new_parametrs["x_lim"][1])/2-width/2,"y":point["y"]-height*2 }, deepcopy(new_parametrs) )

#Класс дерева
class TreeClassifier:

    min_samples_leaf_tree = 50
    class_value = None
    use_predictor_once = True
    getMetric = getGain

    use_auc = True
    use_p_value = True
    use_centroid = True
    use_shap = True
    smooth = 5
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

    root = None

    nodes_on_level = None
    volume = None
    n_leaves = None

    global_rules = None

    def __init__( self,
                          min_samples_leaf_tree = 50, #Минимальное количество объектов необходимое для создания узла (листа) дерева
                          class_value = None, #Класс по которому расчитывается метрика качества разделения (0, 1). Если указано None учитываются оба класса
                          getMetric = getGain, #Ссылка на функцию расчета метрики разделения
                          use_predictor_once = True, #Использовать предиктор в ветви только один раз

                          #Настройки категоризотора
                          use_auc = True, #Использовать пароги Max(AUC)
                          use_p_value = True, #Использовать пароги Min(p_value)
                          use_centroid = True, #Использовать пароги центроиды
                          use_shap = True, #Использовать пароги SHAP

                          smooth = 5, #Степень сглаживания SHAP графика.

                          #Гиперпараметры модели classxgboost.XGBClassifier более подробно описано тут: https://xgboost.readthedocs.io/en/stable/python/python_api.html
                          eval_metric = "auc", learning_rate = None, scale_pos_weight = None, max_depth = None, n_estimators = None,
                          random_state = None, verbosity = None, objective = 'binary:logistic', booster = 'gbtree', tree_method = "exact", max_delta_step = None, gamma = None, min_child_weight = None,
                          subsample = None, colsample_bylevel = None ):

      self.min_samples_leaf_tree = min_samples_leaf_tree
      self.class_value = class_value
      self.use_predictor_once = use_predictor_once
      self.getMetric = getMetric

      self.use_auc = True
      self.use_p_value = True
      self.use_centroid = True
      self.use_shap = True
      self.smooth = 5

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

      #Обучение дерева на основании набора данных.
      # x_data - массив из строк таблицы (массивов) описывающих параметры объекты
      # y_data - массив из строк таблицы (массивов) описывающих ответ решаемой задачи объекты
      # rules - набор правил которые будут использованы при постройке дерева. Если указано None, то будут правила будут расчитаны с помощью класса Сategorizer
 
    def fit(self, x_data, y_data, rules = None, columns_name = None   ):

      if rules == None:
        categorizer = Categorizer(      use_auc = self.use_auc, use_p_value = self.use_p_value, use_centroid = self.use_centroid, use_shap = self.use_shap, smooth = self.smooth,
                                        eval_metric = self.eval_metric , learning_rate = self.learning_rate, scale_pos_weight = self.scale_pos_weight, max_depth = self.max_depth,
                                        n_estimators = self.n_estimators, random_state = self.random_state, verbosity = self.verbosity, objective = self.objective, booster = self.booster,
                                        tree_method = self.tree_method, max_delta_step = self.max_delta_step, gamma = self.gamma,
                                        min_child_weight = self.min_child_weight, subsample = self.subsample, colsample_bylevel = self.colsample_bylevel )
        rules = categorizer.getPointsOfInterest( x_data, y_data, grope_rule = False )

      if columns_name != None:
        ds = pd.DataFrame( x_data, columns=columns_name )
      else:
        ds = pd.DataFrame( x_data )

      y = pd.DataFrame( y_data )
      ds["y"] = y[0]

      self.root = TreeNode( self.min_samples_leaf_tree )

      fit_success = self.root.fit( ds , "y", rules, getMetric = self.getMetric, use_predictor_once = self.use_predictor_once, class_value = self.class_value )

      if not fit_success:
        self.root = TreeLeaf()
        self.root.fit( ds, "y", None, None )
   
    def predict_proba( self, objects ):
      predict = []
      for object_values in objects:

        if not isinstance(object_values, ndarray):
          object_values = [object_values]

        prob = self.root.predict( object_values )
        predict.append( prob )

      return predict

    def get_rules( self, x ):
      self.global_rules = {}
      if( self.root != None ):
        self.root.get_rules( self.global_rules, x=x )

      return self.global_rules

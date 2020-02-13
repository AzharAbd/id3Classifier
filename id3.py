import math
import numpy
import pandas as pd
# from id3 import Id3Estimator
# from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
# from sklearn.tree.export import export_text
# from sklearn.preprocessing import LabelEncoder
# from id3 import export_text as export_text_id3

class Node:
  def __init__(self, conditions):
    self.attrIdx = None
    self.conditions = conditions
    self.children = []
    self.value = None

  def define_attr(self, attrIdx):
    self.attrIdx = attrIdx
    
  def define_value(self, value):
    self.value = value

  def add_children(self, conditions):
    self.children.append(Node(conditions))

  def isLeaf(self):
    return len(self.children) == 0

def printTree(node, count):
  global attribute_names
  global attribute_values
  global target_name

  tab = ''
  for i in range (count):
    tab += '|    '
  tab += '|--- '
  if (node.isLeaf()) :
    print(tab+target_name+ ' == '+str(node.value))
  else :
    count += 1
    for child in node.children:
      print(tab+str(attribute_names[node.attrIdx])+ ' == '+str(child.conditions[node.attrIdx]))
      printTree(child, count)
  
def printRule(node):
  global attribute_names
  global attribute_values
  global target_name

  if (node.isLeaf()) :
    rule = 'IF '
    isFirst = True
    for i in range (len(node.conditions)):
      if not (node.conditions[i] is None):
        if (not isFirst) :
          rule += ' ^ '
        rule += '('+str(attribute_names[i])+' == '+str(node.conditions[i])+')'
        isFirst = False

    print(rule+ ' THEN '+target_name+ ' == '+ str(node.value))
  else :
    for child in node.children:
      printRule(child)

def openCSVFile(filename):
  global data
  global target
  global target_name
  global target_values
  global attribute_names
  global attribute_values

  csv_data = pd.read_csv(filename) 
  
  ##set column names
  attribute_names = csv_data.columns.values[:len(csv_data.columns.values)-1]
  target_name = csv_data.columns.values[len(csv_data.columns.values)-1]
  
  ##set data and target value
  for i in range (csv_data.shape[0]-1):
    attr_data = []
    for j in range (len(attribute_names)) :
      attr_data.append(csv_data[attribute_names[j]][i])
    attr_data.append(csv_data[target_name][i])
    target.append(csv_data[target_name][i])
    data.append(attr_data)

  ##set unique values for each attribute
  for i in range (len(attribute_names)):
    attribute_values.append(list(set(csv_data[attribute_names[i]])))
    available_attribute_idxs.append(i)
  target_values = list(set(csv_data[target_name]))

def openIrisdata():
  global data
  global target
  global target_name
  global target_values
  global attribute_names
  global attribute_values

  iris = load_iris()
  # iris.tolist()

  data = iris.data
  target = iris.target
  attribute_names = iris.feature_names

  ##append target to end of data
  # for i in range(len(data)):
    # print(target[i])
    # data[i] = numpy.append(data[i], target[i])
    # data[i].append(target[i])

  # for i in range(len(data)):
    # print(target[i])
    # print(data[i].tolist().append(target[i]))
    # data[i] = data[i].tolist()


  print(data)
  ##rotate data
  rotated_data = []
  for i in range (len(attribute_names)):
    temp = []
    for j in range (len(data)):
      temp.append(data[j][i])
    rotated_data.append(temp)
  
  ##set unique values for each attribute
  for i in range (len(attribute_names)-1):
    attribute_values.append(list(set(rotated_data[i])))
    available_attribute_idxs.append(i)
  target_values = list(set(target))

def get_valid_data(data, conditions):
  res = []

  for i in range (len(data)):
    is_valid = True
    for j in range (len(conditions)):
      if not (conditions[j] is None):
        if (data[i][j] != conditions[j]) :
          is_valid = False
          break
    if (is_valid):
      res.append(data[i])
  
  return res

def count_entropy(data) :
  value_probability = []
  data_length = len(data)
  for target_value in target_values:
    count = 0
    for row in data:
      if (row[len(row)-1] == target_value):
        count += 1
    value_probability.append(count/data_length)
    
  res = 0
  for value in value_probability:
    if (value != 0):
      res -= value * math.log2(value)

  return res

def most_common_value(data) :
  global target_values

  max_count = 0
  chosen_value = None

  for target_value in target_values :
    count = 0
    for row in data:
      if (row[len(row)-1] == target_value): 
        count+=1
    if (count > max_count) :
      max_count = count
      chosen_value = target_value
  
  return chosen_value
    

def id3(valid_data, available_attribute_idxs, currentNode) :
  global attribute_names
  global attribute_values

  # valid_data = get_valid_data(data, currentNode.conditions)
  global_entropy = count_entropy(valid_data)

  if (len(available_attribute_idxs) and global_entropy) :
    max_entropy = 0
    chosen_attribute_idx = None

    ##iterate all attribute
    for available_attribute_idx in available_attribute_idxs :
      attribute_value = attribute_values[available_attribute_idx]
      temp_entropy = global_entropy

      ##iterate all attribute values
      for value in attribute_value:
        temp_cond = currentNode.conditions[:]
        temp_cond[available_attribute_idx] = value
        attr_valid_data = get_valid_data(valid_data, temp_cond)

        ##count information gain
        if (len(attr_valid_data)):
          temp_entropy -= (len(attr_valid_data)/len(valid_data)) * count_entropy(attr_valid_data)
      
      ##look for maximum information gain
      if (temp_entropy > max_entropy) :
        max_entropy = temp_entropy
        chosen_attribute_idx = available_attribute_idx

      ##reset condition
      temp_cond[available_attribute_idx] = None

    currentNode.define_attr(chosen_attribute_idx)
    children = []

    available_attribute_idxs.remove(chosen_attribute_idx)

    ##construct node children
    temp_cond = currentNode.conditions[:]
    for value in attribute_values[chosen_attribute_idx]:
      temp_cond[chosen_attribute_idx] = value
      next_valid_data = get_valid_data(valid_data, temp_cond)
      if (len(next_valid_data)):
        currentNode.add_children(temp_cond[:])
        id3(next_valid_data, available_attribute_idxs, currentNode.children[len(currentNode.children)-1])  
    # available_attribute_idxs.remove(chosen_attribute_idx)

    # for child in currentNode.children :
    #   id3(valid_data, available_attribute_idxs, child)
  else :
    currentNode.value = most_common_value(valid_data)

data = []
target = []
target_name = 'target'
target_values = []
attribute_names = []
attribute_values = []
available_attribute_idxs = []

##Lod data
# print("Input filename")
# filename = input()
# openCSVFile(filename)
openIrisdata()

##Set default variables
conditions = []
for i in range (len(attribute_names)):
  conditions.append(None)
root = Node(conditions)

##Execute id3
id3(data, available_attribute_idxs, root)
printTree(root, 0)
printRule(root)

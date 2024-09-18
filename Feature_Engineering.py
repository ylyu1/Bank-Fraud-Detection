import numpy as np
import pandas as pd

# class Ordinal_Transformer_lob():


#     def fit(self, X, y=None):
#         self.res = {"bc": 1, np.nan: 0}

#     def transform(self, X, y=None):
#         list_a=[0]*len(X)
#         for i in range(0,len(X)):
#             list_a[i] = self.res[X[i]]
#         return pd.DataFrame(list_a,columns=['Sender_lob_conv'])

#     def fit_transform(self, X, y=None):
#         self.fit(X, y)
#         return self.transform(X, y)

class Ordinal_Transformer_lob():

    
    def fit(self, X, y=None):
        self.lob_value = X.unique()[0]  

    def transform(self, X, y=None):
        # Create a new column 'Sen_lob_conv' with 1 for "bc" or mask value and 0 for other values
        Sen_lob_conv = (X == self.lob_value).astype(int)
        return Sen_lob_conv
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
class Frequency_Transformer_Single():

    
    def __init__(self, name_f, name_Label):
        self.name_f = name_f
        self.Label = name_Label
        self.nf_name = name_f+"_ave"
        
    def fit(self, X, y=None):
        self.group = X.groupby(by = self.name_f)[self.Label].mean()
        self.med = np.nanmean(self.group.values)
 
        
    def transform(self, X, y=None):
        X[self.nf_name] = X[self.name_f].map(self.group)
        if X[self.nf_name].isna().sum()==0:
            return X
        else:
            if X[self.name_f].dtype=='float64':
                return self.numeric_fill(X, y)
            else:
                X[self.nf_name].fillna(self.med, inplace=True)
                return X
  #      X[self.nf_name]=np.log((X[self.nf_name]-0.00001)/(1-X[self.nf_name]+0.00001))       
               
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def numeric_fill(self, X, y=None):
        nan_indices = X[self.nf_name].isna()
        values_in_feature2 = X.loc[nan_indices, self.name_f]
        list_a=[]
        num = 10
        for x in values_in_feature2.values:
            list_a.append(self.find_near(x,num))
        
        X[self.nf_name][nan_indices]=list_a
        return X
        
    def find_near(self, x, num):
        A_group = np.array(self.group.index)
        diff = np.abs(A_group-x)
        indices = np.argpartition(diff,num)[:num]
        nearest_values=A_group[indices]
        x = self.group[nearest_values].sum()/num
        return x

class Feq_Transformer_Multi():

    
    def __init__(self, names, label):
        self.transformers = []
        for name in names:
            fq = Frequency_Transformer_Single(name, label)
            self.transformers.append(fq)        
        
    def fit(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.fit(X)
        return X
            
    def transform(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.transform(X)
        return X

    def fit_transform(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.fit_transform(X)
        return X
            
    def numeric_fill(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.numeric_fill(X)
        return X
    
    def find_near(self, X, num):
        for transformer in self.transformers:
            X = transformer.find_near(X)
        return X
        
class Dummy_Transformer(object):

    def fit(self, X, y=None):
        self.keys = set(X)

    def transform(self, X, y=None):
        res = {}
        for key in self.keys:
            res[key] = [0]*len(X)
        for i, item in enumerate(X):
            if item in self.keys:
                res[item][i] = 1
        return pd.DataFrame(res)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class Pair_Transformer():

    
    def __init__(self, names:list[tuple()]):
        self.names=names
        
    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        for name in self.names:
            new_column = (name[0] + name[1] + 'Pair').replace('Sen_CtrBen_Ctr', 'SBC').replace('Sen_Iddoy', 'SDAY').replace('USD_ordinalTT', 'USD_TT_')
            X[new_column] = X[name[0]] + '-' + X[name[1]].astype('str')
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

# create a tranformer for Sen/Ben filling missing to each other
class Fill_Missing_Transformer(object):

    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Sen_Id'] = X['Sen_Id'].fillna(X['Ben_Id'])
        X['Ben_Id'] = X['Ben_Id'].fillna(X['Sen_Id'])

        X['Sen_Act'] = X['Sen_Act'].fillna(X['Ben_Act'])
        X['Ben_Act'] = X['Ben_Act'].fillna(X['Sen_Act'])

        X['Sen_Ctr'] = X['Sen_Ctr'].fillna(X['Ben_Ctr'])
        X['Ben_Ctr'] = X['Ben_Ctr'].fillna(X['Sen_Ctr'])

        X['Sen_Sec'] = X['Sen_Sec'].fillna('Unknown').astype('str')
        
        return X  # Return the modified DataFrame

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class Time_Transformer(object):

    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Assuming the datetime column is named 'Time_step'
        X['Time_step'] = pd.to_datetime(X['Time_step'])
        X['date'] = X['Time_step'].dt.date
        X['doy'] = X['Time_step'].dt.doy
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)



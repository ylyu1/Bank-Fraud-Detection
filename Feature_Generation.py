## A couple of functions of generating new features
import numpy as np
import pandas as pd

class Location_Transformer():
    
    def fit(self, X, y=None):
        file_path = '../data/CP.csv'  
        cp = pd.read_csv(file_path)
        cp['name'] = cp['name'].str.lower()

        ##Create two dictionaries for looking up
        self.dict_la = {key: value for key, value in zip(cp['name'], cp['latitude'])}
        self.dict_lg = {key: value for key, value in zip(cp['name'], cp['longitude'])}       
        
    # def Loc_Cor_transform(self, X):
    # add transform
    def transform(self, X, y=None):
        # change function name from Loc_Cor_transform to transform
        X['B_Ctr'] = X.B_Ctr.str.replace('-', ' ')
        X['S_Ctr'] = X.S_Ctr.str.replace('-', ' ')
        df_sc=self.Sen_transform(X.S_Ctr)
        df_bc=self.Ben_transform(X.B_Ctr)
        X = pd.concat([X, df_sc, df_bc], axis=1)
        X = self.haversine(X)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def Ben_transform(self, X_Ben):
        list_a=[]
        list_b=[]
        for x in X_Ben:
            if x.lower() in self.dict_la:
                list_a.append(self.dict_la[x.lower()])
                list_b.append(self.dict_lg[x.lower()])
            else:
                list_a.append(0)
                list_b.append(0)
        return pd.DataFrame({'Ben_lat': list_a, 'Ben_lg': list_b})
    
    def Sen_transform(self, X_Sen):
        list_a=[]
        list_b=[]
        for x in X_Sen:
            if x.lower() in self.dict_la:
                list_a.append(self.dict_la[x.lower()])
                list_b.append(self.dict_lg[x.lower()])
            else:
                list_a.append(0)
                list_b.append(0)
        return pd.DataFrame({'Sen_lat': list_a, 'Sen_lg': list_b})
      
    def haversine(self, X):
        # Convert latitude and longitude from degrees to radians
        # Convert decimal degrees to radians 
        X['lat1'], X['lon1'], X['lat2'], X['lon2'] = map(np.radians, [X.Sen_lat, X.Sen_lg, X.Ben_lat, X.Ben_lg])

        # Haversine formula 
        X['dlon'] = X['lon2'] - X['lon1'] 
        X['dlat'] = X['lat2'] - X['lat1'] 

        X['a'] = np.sin(X['dlat'] / 2)**2 + np.cos(X['lat1']) * np.cos(X['lat2'])*np.sin(X['dlon'] / 2)**2
        X['c'] = 2 * np.arctan2(np.sqrt(X['a']), np.sqrt(1 - X['a'])) 

        # Radius of earth in kilometers. Use 3956 for miles
        r = 6371 
        X['dist'] = r * X['c']
        cols_to_drop = ['lat1','lon1','lat2','lon2','dlon','dlat','a','c']
        X.drop(columns=cols_to_drop,inplace = True)
        return X
    
class Group_Transformer():
    
    def __init__(self, binwidth, fn):
        self.bin = binwidth
        self.fn = fn
        
    def fit(self, X, y=None):
        pass
        
    def transform(self, X, y=None):
        # Generate interval features for numerical values
        bin_edges = list(range(0, int(X[self.fn].max()) + self.bin, self.bin))
        ng = self.fn+'_group'
        X[ng] = pd.cut(X[self.fn], bins=bin_edges, right=False, include_lowest=True)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

# convert individual_Behavior function to class
class Individual_Behavior_Transformer():
    
    def __init__(self, names:list[str]):
        self.names=names
        
    def fit(self, X, y=None):
        pass
        
    def transform(self, X, y=None):
        for name in self.names:
            X = self.individual_behavior(X, name)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def individual_behavior(self, X, name):
        # name can be "Sen_Id", "Ben_Id"
        # Count trans per day for each Sen_id
        count_name = name+'_dc'
        daily_trans = X.groupby([name, 'date']).size().reset_index(name=count_name)

        # Merge back to the original dataframe
        X = X.merge(daily_trans[[name, 'date', count_name]], on=[name, 'date'], how='left')

        # Sort by Sen_id and date
        daily_trans.sort_values(by=[name, 'date'], inplace=True)

        # Calculate the difference in daily transaction counts
        count_diff_name = name+'_cdf'
        increase_rate_name = name+'_incr'
        daily_trans[count_diff_name] = daily_trans.groupby(name)[count_name].diff().fillna(0)
        daily_trans[increase_rate_name] = daily_trans.groupby(name)[count_name].pct_change().fillna(0)
        
        # Merge back with original DataFrame
        X = X.merge(daily_trans[[name, 'date', count_diff_name, increase_rate_name]], on=[name, 'date'], how='left')

        # Calculate the cumulative average up to the day before each transaction
        daily_trans['cumulative_sum'] = daily_trans.groupby(name)[count_name].cumsum().shift(1)
        daily_trans['transaction_days'] = daily_trans.groupby(name).cumcount()

        cum_avg_name=name+'_cumavg'
        daily_trans[cum_avg_name] = daily_trans['cumulative_sum'] / daily_trans['transaction_days']
        daily_trans[cum_avg_name] = daily_trans[cum_avg_name].fillna(0)
        daily_trans[cum_avg_name] = daily_trans[cum_avg_name].replace([-np.inf, np.inf], 0)
        
        # Calculate the difference
        avg_diff_name=name+'_avgdif'
        daily_trans[avg_diff_name] = daily_trans[count_name]-daily_trans[cum_avg_name].fillna(0)
        daily_trans[avg_diff_name] = daily_trans[avg_diff_name].replace([-np.inf, np.inf], 0)

        # Calculate the difference
        avg_incr_name=name+'_avgincr'
        daily_trans[avg_incr_name] = ((daily_trans[count_name] -   daily_trans[cum_avg_name])/daily_trans[cum_avg_name]).fillna(0)
        daily_trans[avg_incr_name] = daily_trans[avg_incr_name].replace([-np.inf, np.inf], 0)

        # Calculate days since initial
        initial_dates = daily_trans.groupby(name)['date'].min().rename('initial_date')

        # Merge the initial transaction date back into the DataFrame
        daily_trans = daily_trans.merge(initial_dates, on=name, how='left')

        # Calculate the number of days since the initial transaction
        day_sin_name=name+'_daysin'
        daily_trans[day_sin_name] = (daily_trans['date'] - daily_trans['initial_date']).dt.days

        # Merge back with original DataFrame
        X['date'] = pd.to_datetime(X['date'])  # Ensure 'date' format is consistent
        daily_trans['date'] = pd.to_datetime(daily_trans['date'])

        X = X.merge(daily_trans[[name, 'date', avg_diff_name]], on=[name, 'date'], how='left')

        X = X.merge(daily_trans[[name, 'date', avg_incr_name]], on=[name, 'date'], how='left')

        X = X.merge(daily_trans[[name, 'date', day_sin_name]], on=[name, 'date'], how='left')

        X = X.merge(daily_trans[[name, 'date', cum_avg_name]], on=[name, 'date'], how='left')
        return X
    
# convert time_difference function to class
class Time_Difference_Transformer():
    
    def __init__(self, names:list[str]):
        self.names = names  
        
    def fit(self, X, y=None):
        pass    
        
    def transform(self, X, y=None):
        for name in self.names:
            X = self.time_difference(X, name)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
   
    def time_difference(self, X, name):
        X.sort_values(by=[name,'Time_step'], inplace=True)
        time_diff_name = name+'_timediff'
        X['shifted_time'] = X.groupby(name)['Time_step'].shift()
        X[time_diff_name] = (X['Time_step']-X['shifted_time']).fillna(pd.Timedelta(seconds=0))
      #  df[time_diff_name] = df.groupby(name)['Time_step'].diff().fillna(pd.Timedelta(seconds=0)) 
        X[time_diff_name] = X[time_diff_name].dt.total_seconds()/60.0
        X.drop(['shifted_time'], axis=1, inplace=True)
        return X
    
# convert Geo_velocity to class
class Geo_Velocity_Transformer():  
    
    def __init__(self, names:list[str]):
        self.names = names  
        
    def fit(self, X, y=None):
        pass    
        
    def transform(self, X, y=None):
        for name in self.names:
            X = self.geo_velocity(X, name)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)          
        
    # def Sen/Ben_geo_velocity(df,name):
    def geo_velocity(self, X, name):   
        # Sort the DataFrame by 'Sen_id' and 'time_stamp'
        X.sort_values(by=[name + '_Id', 'Time_step'], inplace=True)

        # Shift the latitude and longitude columns
        X['shifted_lon_sac'] = X.groupby(name + '_Id')[name + '_lg'].shift()
        X['shifted_lat_sac'] = X.groupby(name + '_Id')[name + '_lat'].shift()

        # Apply the vectorized haversine function
        geo_dist_name=name + '_Id' + '_geodist'
        X[geo_dist_name] = self.haversine_vectorized(X['shifted_lon_sac'], X['shifted_lat_sac'], X[name + '_lg'], X[name + '_lat'])

        # Drop the shifted columns if they are no longer needed
        X.drop(['shifted_lon_sac', 'shifted_lat_sac'], axis=1, inplace=True)
        return X               
        
    def haversine_vectorized(self, lon1, lat1, lon2, lat2):
        # Convert latitude and longitude from degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers. Use 3956 for miles
        return c * r

# create a tranformer
class USD_Ordinal_Transformer(object):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['USD_ordinal'] = X['amount'].map(self.USD_Transform_ordinal)
        return X  # Return the modified DataFrame

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def USD_Transform_ordinal(self, x):
        # The provided transformation logic
        if x >= 9000:
            return 'high'
        elif x >= 100 and x < 1000:
            return 'medium'
        else:
            return 'low'

class Feature_Selection_Transformer(object):
    def __init__(self, selected_features: list[str]):
        self.selected_features = selected_features
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        existing_features = [feature for feature in self.selected_features if feature in X.columns]
        return X[existing_features]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)












    
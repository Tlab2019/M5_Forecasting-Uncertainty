import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from fbprophet import Prophet

class myProphet():
    
    def __init__(self,holiday=None,countory=None):
        
        self.model = Prophet(daily_seasonality=True)
        
        if holiday is not None:
            self.model = Prophet(daily_seasonality=True,
                                 holiday=holiday)
        if countory is not None:
            self.model = self.model.add_country_holidays(country_name=countory)
        
        
    def fit(self,y):
        
        self.df = pd.DataFrame()
        self.df["ds"] = y.index
        self.df["y"] = y.values
        
        self.model.fit(self.df)
        
        
        return self.model
        
    def predict(self,start,end,freq,mode="uncertainty"):
        
        gap = (end - start).days
        
        future = pd.DataFrame(data=pd.date_range(start,end,freq=freq),columns=["ds"])
        forecast = self.model.predict(future)
        #print(forecast)
        
        pred = pd.Series(forecast["yhat"].values,
                         index=pd.to_datetime(forecast["ds"]))
        
        output = pred
        
        if mode == "uncertainty":
            unc = pd.DataFrame(index=pd.to_datetime(forecast["ds"]))
            unc["lower"] = forecast['yhat_lower'].values
            unc["upper"] = forecast['yhat_upper'].values
            
            output = [pred,unc]
        elif mode == "all":
            output = forecast.set_index("ds",drop=True)
        else:
            pass
            
        return output
    
    
    
if __name__ == "__main__":
    
    x = np.linspace(0,2,100)
    time = pd.date_range(start="2020-01-01",end="2020-01-31",freq="0.5H")
    x = np.linspace(0,100,len(time))
    data = pd.Series(np.sin(x),index=time)
    
    
    model = myProphet(y=data)
    
    model.fit()
    
    import datetime as dt
    pred = model.predict(start=dt.datetime(2020,1,31),end=dt.datetime(2020,2,8),freq="0.5H")
    
    print(data)
    
    pd.plotting.register_matplotlib_converters()
    data.plot()
    pred.plot(label="prediction",color="red")
    plt.grid(True)
    plt.show()
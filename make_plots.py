from itertools import count
from sqlite3 import Date
from statistics import mean
from string import whitespace
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
import seaborn as sns 

# Read in bikes.csv into a pandas dataframe
### Your code here
df = pd.read_csv('bikes.csv')
# Read in DOX.csv into a pandas dataframe
# Be sure to parse the 'Date' column as a datetime
### Your code here
df1 = pd.read_csv('DOX.csv',parse_dates=['Date'])
# Divide the figure into six subplots
# Divide the figure into subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 12))


# Make a pie chart
### Your code here

l=df['status'].unique()
gp=df.groupby(['status']).size()
axs[0,0].pie(gp,labels=l,autopct='%1.0f%%')
axs[0,0].set_title('Current Status')

# Make a histogram with quartile lines
# There should be 20 bins
### Your code here
nbin = 20
axs[0,1].hist(df['purchase_price'],bins = nbin,color = 'skyblue' )
axs[0,1].set_xlabel('US dollars')
axs[0,1].set_ylabel('Number Of Bikes')
axs[0,1].set_title('Price Histogram (' + str(len(df)) + " bikes)")
axs[0,1].axvline(min(df['purchase_price']),linewidth = 2, linestyle ="--",color ='black')
axs[0,1].text(min(df['purchase_price'] +11) ,10,"Min: "+str(min(df['purchase_price'])),rotation=90)
axs[0,1].axvline(df['purchase_price'].quantile(0.25),linewidth = 2, linestyle ="--",color ='black')
axs[0,1].text(df['purchase_price'].quantile(0.25)+11,10,"25%: " + str(round(df['purchase_price'].quantile(0.25))),rotation=90)
axs[0,1].axvline(df['purchase_price'].quantile(0.50),linewidth = 2, linestyle ="--",color ='black')
axs[0,1].text(df['purchase_price'].quantile(0.50)+11,10,"50%: " + str(round(df['purchase_price'].quantile(0.50))),rotation=90)
axs[0,1].axvline(df['purchase_price'].quantile(0.75),linewidth = 2, linestyle ="--",color ='black')
axs[0,1].text(df['purchase_price'].quantile(0.75)+11,10,"75%: " + str(round(df['purchase_price'].quantile(0.70))),rotation=90)
axs[0,1].axvline(max(df['purchase_price']),linewidth = 2, linestyle ="--",color ='black')
axs[0,1].text(max(df['purchase_price'])+11,10,"Max :" + str(round(max(df['purchase_price']))),rotation=90)

# Make a scatter plot with a trend line
### Your code here
x=df['purchase_price'].values.reshape(-1,1)
y=df['weight'].values.reshape(-1,1)
axs[1,0].scatter(x,y,s=5,color= 'grey')
axs[1,0].set_xlabel('Price')
axs[1,0].set_ylabel('Weight')
axs[1,0].set_title('Price vs Weight')

# Get data as numpy arrays
X = df['purchase_price'].values.reshape(-1, 1)
y = df['weight'].values.reshape(-1, 1)
yl = axs[1,0].get_yticks()
xl = axs[1,0].get_xticks()
axs[1,0].set_yticks(yl)
axs[1,0].set_xticks(xl)

axs[1,0].set_yticklabels([ str(y) + 'kg' for y in yl])
axs[1,0].set_xticklabels([ '$' +str(x)  for x in xl])

# Do linear regression
reg = LinearRegression()
reg.fit(X, y)
y_pred=reg.predict(x)
axs[1,0].plot(x,y_pred,color='red')
# Get the parameters
slope = reg.coef_[0]
intercept = reg.intercept_
print(f"Slope: {slope}, Intercept: {intercept}")


#dox
d1 = pd.DataFrame(df1, columns = ['Date','Adj Close'])
d1['Date']=pd.to_datetime(d1['Date'])
d1.set_index('Date',inplace=True)
axs[1,1].plot(d1)
axs[1,1].set_title('DOX')
axs[1,1].grid(True)
yl = axs[1,1].get_yticks()
axs[1,1].set_yticks(yl)
axs[1,1].set_yticklabels(['$' + str(y) for y in yl])
xl = axs[1,1].get_xticks()






# Make a boxplot sorted so mean values are increasing
# Hide outliers
### Your code here

g = df.groupby(['brand'])
dfb = pd.DataFrame({col: val['purchase_price'] for col, val in g})
median = dfb.median()
median.sort_values(inplace=True)
dfb = dfb[median.index]
dfb.boxplot(ax=axs[2,0], showfliers = False)
yl = axs[2,0].get_yticks()
axs[2,0].set_yticks(yl)
axs[2,0].set_yticklabels(['$' + str(y) for y in yl])
axs[2,0].set_title('Brand vs Price')


# Make a violin plot
### Your code here
l=[]
for x in dfb.keys():
    l.append(df.query('brand == @x')['purchase_price'])
axs[2,1].set_xticks([1,2,3,4,5,6])
axs[2,1].set_xticklabels(dfb.keys())
axs[2,1].violinplot(l,showmedians=True)
yl = axs[2,1].get_yticks()
axs[2,1].set_yticks(yl)
axs[2,1].set_yticklabels(['$' + str(y) for y in yl])
axs[2,1].set_title('Brand vs Price')
# Create some space between subplots

plt.subplots_adjust(left=0.125,
                    bottom=.11, 
                    right=0.871, 
                    top=0.88, 
                    wspace=.669, 
                    hspace=.46)

# Write out the plots as an image
plt.savefig('plots.png')
plt.show()

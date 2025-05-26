import pandas as pd
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
year= True

main = ["Year","IsRookieIndep","IsRookieNonIndep","IsNonRookieIndep","IsNonRookieNonIndep"]
start_date = '2013-01-01'
end_date = '2020-12-31'

df= pd.read_csv("car.csv")
df["Appointment Date"]= pd.to_datetime(df["Appointment Date"],format= "mixed")
df["Year"]= df["Appointment Date"].dt.year

df = df[(df['Appointment Date'] >= start_date) & (df['Appointment Date'] <= end_date)]

col = pd.read_csv("cols.csv")
col = col["col"].to_list()

#Main Function for the T-Test and P Value
def p_values(frame):
    categories = df2['Category'].unique()
    d1 = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            cat1 = categories[i]
            cat2 = categories[j]
            group1 = df2[df2['Category'] == cat1][car]
            group2 = df2[df2['Category'] == cat2][car]
            
            t_stat, p_value = ttest_ind(group1, group2)
            d1.append((cat1, cat2, t_stat, p_value))
    
    d1= pd.DataFrame(d1)
    
    new_col = ["R_IND_NIND","IND_NR_R","NA" ,"NA2" ,"NIND_NR_R","NR_IND_NIND"]
    new_col2 = ["R_IND_NIND","IND_NR_R","NIND_NR_R","NR_IND_NIND"]
    
    d1.iloc[:,0]= new_col
    d1 = d1[[0,3]]
    d1 = d1.set_index(0).T[new_col2]
    d1= d1.T.rename(columns = {3:"p_val"}).T
    
    p_values= []
    for i in categories :
        x= df2[df2["Category"]== i][car]
        t_stat, p_val = ttest_1samp(x,0)
        p_values.append((i,p_val))
    
    d2= pd.DataFrame(p_values).set_index(0)
    d2= d2.rename(columns = {1:"p_val"})
    d2= d2.T
    
    p_val = pd.concat([d2,d1], axis =1)
    return p_val

result= pd.DataFrame()

for car in col:
    try:
        
        df2= df[[car] + main]
        df2= df2.dropna(subset= car)
        df2["Category"]= df2.loc[:,"IsRookieIndep":].idxmax(axis=1)
        if year:
            df3= df2.groupby(["Year","Category"])[car].mean(0).reset_index()
            df3= df3.pivot(index="Year",columns="Category",values= car).reset_index()
            df3= df3.set_index("Year")
            df3["IND_NR_R"]=df3["IsNonRookieIndep"] -df3["IsRookieIndep"]
            df3["NIND_NR_R"]= df3["IsNonRookieNonIndep"] - df3["IsRookieNonIndep"]
            df3["R_IND_NIND"]= df3["IsRookieIndep"] - df3["IsRookieNonIndep"]
            df3["NR_IND_NIND"]= df3["IsNonRookieIndep"] - df3["IsNonRookieNonIndep"]
            pval = df3.apply(lambda x: ttest_1samp(x.dropna(), 0)).iloc[1,:].to_frame("p_val").T
            mean = df3.mean().to_frame("mean").T
            df4= pd.concat([mean,pval],axis=0).T
            df4["carid"] =car
        
        else:
            df3 = df2.groupby("Category")[car].mean().to_frame("mean").T
            df3["IND_NR_R"]=df3["IsNonRookieIndep"] -df3["IsRookieIndep"]
            df3["NIND_NR_R"]= df3["IsNonRookieNonIndep"] - df3["IsRookieNonIndep"]
            df3["R_IND_NIND"]= df3["IsRookieIndep"] - df3["IsRookieNonIndep"]
            df3["NR_IND_NIND"]= df3["IsNonRookieIndep"] - df3["IsNonRookieNonIndep"]
            
            p_val= p_values(df2).T
            
            df4 = df3.T.join(p_val).reset_index()
            df4["carid"]= car
    except:
        continue
    else:
        result= pd.concat([result,df4],axis=0)

if year:
    name = "fixedef_yr"
else:
    name= "ind"

result.to_csv("result_" + name + ".csv")
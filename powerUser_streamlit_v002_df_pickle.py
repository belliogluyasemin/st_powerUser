import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.model_selection import train_test_split
import shap
from datetime import date, datetime




df = pd.read_pickle("df.pkl")

country_filtered = df[df['country'].isin(["United States", "Canada", "Germany", "India", "France", "Taiwan",'Italy', "Japan","Spain", "China", "Singapore", "South Korea", "Netherlands", 'Turkey'])]
all_filtered = country_filtered.drop(['itemRevenue2','itemRevenue90', 'user_pseudo_id','name','source','cnt','fdate'], axis = 1)



# Page Configuration
st.set_page_config(
    page_title="Power User",
    page_icon= "powerUserIcon.png",

)

st.markdown("""
    <h1 style='color: #876C6C;'>Prediction of Power User </h1>
    """, unsafe_allow_html=True)
##st.title("Prediction of Power User ")
st.subheader("About Data")
st.markdown("""
 [Google Merchandise Store](https://shop.merch.google/) is an online store that sells Google-branded merchandise. The site uses Google Analytics 4's standard web ecommerce implementation along with enhanced measurement. The ga4_obfuscated_sample_ecommerce dataset available through the [BigQuery Public Datasets](https://console.cloud.google.com/bigquery) program contains a sample of obfuscated BigQuery event export data for three months from 2020-11-01 to 2021-01-31.
""")
st.header("Data From BigQuery as df")
st.write(df.head(30))
st.header("Feature's")
st.markdown("""

- **sessionCnt**: The number of sessions conducted by a user within a specified period.
- **sessionDate**: The dates when the user started a session.
- **itemBrandCount**: The number of different brands viewed by users within a specified period.
- **itemCatCount**: The number of different product categories viewed by users.
- **viewPromotion**: The number of times promotions were viewed.
- **selectPromotion**: The number of promotions selected by users.
- **itemViewCnt**: The number of times products were viewed.
- **itemSelectCnt**: The number of times products were selected.
- **paymetInfoAdd**: The number of times payment information was added.
- **shippingInfoAdd**: The number of times shipping information was added.
- **ScrollpageLocationCnt**: The count of unique page locations where users performed a scroll action.
- **ScrollpageTitleCnt**: The count of unique page titles where users performed a scroll action.
- **pageViewPageLocationCnt**: The number of visits to different page locations.
- **pageViewPageTitleCnt**: The number of visits to different page titles.
- **addToCarts**: The number of times products were added to the cart.
- **checkOut**: The number of times users initiated the checkout process.
- **ecommercePurchases**: The number of purchase transactions made through the e-commerce platform.
- **purchaseToViewRate**: The purchase rate relative to product views (expressed as a percentage).
- **itemPurchaseName**: The number of distinct product names purchased.
- **itemPurchaseQuantity**: The total quantity of products purchased.
- ***itemRevenue15***: Revenue from products calculated using a different method.
""")

##Side Bar

st.sidebar.markdown("**Enter & Select  User Metric Below**")

medium_options = ['organic', '(none)', 'cpc', 'referral', '<Other>', '(data deleted)']
medium = st.sidebar.selectbox("Select Medium:", medium_options, help='organic, (none), cpc, referral, <Other>, (data deleted)')

mobile_brand_options = ['Samsung', 'Microsoft', 'Google', 'Apple', '<Other>']
mobile_brand_name = st.sidebar.selectbox("Select Mobile Brand Name:", mobile_brand_options, help='Samsung, Microsoft, Google, Apple, <Other>')

country_options = ['United States', 'Canada', 'Germany', 'India', 'France', 'Taiwan', 'Italy', 'Japan', 'Spain', 'China', 'Singapore', 'South Korea', 'Netherlands', 'Turkey']
country = st.sidebar.selectbox("Select Country:", country_options, help='United States, Canada, Germany, India, France, Taiwan, Italy, Japan, Spain, China, Singapore, South Korea, Netherlands, Turkey')

category_options = ['mobile', 'desktop', 'tablet']
category = st.sidebar.selectbox("Select Category:", category_options, help='mobile, desktop, tablet')

sessionCnt = st.sidebar.number_input("First 15 Day Session Count", min_value=0 , step=1)
sessionDate = st.sidebar.number_input("First 15 Day Session Date", min_value=0 , step=1)
itemBrandCount = st.sidebar.number_input("First 15 Day Diffirent Item Brands", min_value=0 , step=1)
itemCatCount = st.sidebar.number_input("First 15 Day Diffirent Item Category", min_value=0 , step=1)
viwePromotion = st.sidebar.number_input("First 15 Day Promotion Viewed ", min_value=0 , step=1)
SelectPromotion = st.sidebar.number_input("First 15 Day Promotion Selected ", min_value=0 , step=1)
itemViewCnt = st.sidebar.number_input("First 15 Day Item View Count ", min_value=0, step=1)
itemSelectCnt = st.sidebar.number_input("First 15 Day item Select Conut ", min_value=0 , step=1)
paymetInfoAdd = st.sidebar.number_input("First 15 Day Added Payment Info Count ", min_value=0, step=1)
shippingInfoAdd = st.sidebar.number_input("First 15 Day Adding Shipping Info ", min_value=0, step=1)
ScrollpageLocationCnt = st.sidebar.number_input("First 15 Day Scrolling Different Page Location", min_value=0, step=1)
ScrollpageTitleCnt = st.sidebar.number_input("First 15 Day Scrolling Different Page Title", min_value=0, step=1)
pageViewPageLocationCnt = st.sidebar.number_input("First 15 Day Viewed Different Page location", min_value=0, step=1)
pageViewPageTitleCnt = st.sidebar.number_input("First 15 Day Viewed Different Page Title", min_value=0, step=1)
itemViews = st.sidebar.number_input("First 15 Day Item Viewed", min_value=0, step=1)
addToCarts = st.sidebar.number_input("First 15 Day Item Added to Basket", min_value=0, step=1)
addToItemId = st.sidebar.number_input("First 15 Day Added to Basket Different Item", min_value=0, step=1)
searchResultViewedCnt = st.sidebar.number_input("First 15 Day Seacrh REsuls Viewed Count", min_value=0, step=1)
checkOut = st.sidebar.number_input("First 15 Day checkOut Count", min_value=0, step=1)
ecommercePurchases = st.sidebar.number_input("First 15 Day Purchase Count", min_value=0, step=1)
purchaseToViewRate = st.sidebar.number_input("First 15 Day Item View to Purchase Rate", min_value=0, max_value= 1)
itemPurchaseName = st.sidebar.number_input("First 15 Day Added Different Purchased Item", min_value=0, step=1)
itemPurchaseQuantity = st.sidebar.number_input("First 15 Day Added Purchase Quantity", min_value=0, step=1)
itemRevenue15 = st.sidebar.number_input("First 15 Day Revenue ($)", min_value=0)



rf100Model = load("randomForest100Model_new.pkl")

input_df = pd.DataFrame({
'medium': [medium],
'mobile_brand_name': [mobile_brand_name],
'country' :[country],
'category' : [category],
'sessionCnt': [sessionCnt],
'sessionDate':[sessionDate],
'itemBrandCount':[itemBrandCount],
'itemCatCount':[itemCatCount],
'viwePromotion':[viwePromotion],
'SelectPromotion':[SelectPromotion],
'itemViewCnt'	:[itemViewCnt],
'itemSelectCnt':[itemSelectCnt],
'paymetInfoAdd':[paymetInfoAdd],
'shippingInfoAdd':[shippingInfoAdd],
'ScrollpageLocationCnt':[ScrollpageLocationCnt],
'ScrollpageTitleCnt':[ScrollpageTitleCnt],
'pageViewPageLocationCnt':[pageViewPageLocationCnt],
'pageViewPageTitleCnt':[pageViewPageTitleCnt],
'itemViews':[itemViews],
'addToCarts':[addToCarts],
'addToItemId':[addToItemId],
'searchResultViewedCnt':[searchResultViewedCnt],
'checkOut':[checkOut],
'ecommercePurchases':[ecommercePurchases],
'purchaseToViewRate':[purchaseToViewRate],
'itemPurchaseName':[itemPurchaseName],
'itemPurchaseQuantity':[itemPurchaseQuantity],
'itemRevenue15':[itemRevenue15]
})


new_df = pd.concat([all_filtered,input_df], axis = 0)

gdp = pd.read_pickle("gdp.pkl")
gdp = gdp[['GDP per capita, current prices\n (U.S. dollars per capita)','2020','2021']]
gdp = gdp.rename(columns= {'GDP per capita, current prices\n (U.S. dollars per capita)':'country'})
gdp = gdp.rename(columns= {'2020':'gdp_2020_value'})
gdp = gdp.rename(columns= {'2021':'gdp_2021_value'})
merged_df = pd.merge(new_df, gdp, on='country',  how='left')
merged_df['gdp_2020_value'] = merged_df['gdp_2020_value'].astype(float)
merged_df['gdp_2021_value'] = merged_df['gdp_2021_value'].astype(float)
merged_df['Avg_gdp'] =  merged_df[['gdp_2020_value','gdp_2021_value']].mean(axis = 1)
merged_df.dropna(inplace = True)

merged_df.loc[:, 'LogGDP'] = np.log(merged_df['Avg_gdp'])
dummies_input = pd.get_dummies(merged_df[['medium', 'mobile_brand_name', 'country', 'category']], drop_first=True, dtype=int)
dummies_input = pd.concat([merged_df,dummies_input], axis = 1)
x_input = dummies_input.drop(['medium','mobile_brand_name','country','category'], axis = 1,)
xc_input =  x_input.copy()
xc_input['perBasket'] = xc_input.apply(
    lambda row: row['itemPurchaseQuantity'] / row['ecommercePurchases'] if row['ecommercePurchases'] != 0 else 0,
    axis=1
)
##xc_input = pd.DataFrame(xc_input)
xc_input = xc_input.rename(columns={"medium_(none)": "medium_none", "medium_<Other>": "medium_Other"})


test_df = x_input.iloc[-1].to_frame().T
test_df_c = xc_input.iloc[-1].to_frame().T

pred_ltv90 = np.round(np.exp(rf100Model.predict(test_df)),2)

xgb_adasyn_model = load("xgb_adasyn_model.pkl")
pred_class = xgb_adasyn_model.predict(test_df_c)
prob_class = np.round(xgb_adasyn_model.predict_proba(test_df_c),2)


st.header("Results")
st.markdown("When you enter the data on the left side, resuls will be listed below.")
# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'Date': [today],
    'Time': [time],
    'Predicted LTV90': [pred_ltv90],
    'Prediction of Class': [pred_class],
    'Power User Prob': [prob_class[:,1]],
    'Not Power User Prob': [prob_class[:,0]]

    })
    results_df["Prediction of Class"] = results_df["Prediction of Class"].apply(lambda x: str(x).replace('[0]',"Not Power User"))
    results_df["Prediction of Class"] = results_df["Prediction of Class"].apply(lambda x: str(x).replace('[1]',"Power User"))
    

    st.table(results_df)



    explainer = shap.Explainer(xgb_adasyn_model)
    shap_values_class = explainer(test_df_c)


    st.header("SHAP Local Waterfall Plot")
    st.markdown("""
        The SHAP waterfall plot below visualizes the impact of each feature on the model's prediction for the specific instance provided.
    """)

    # SHAP waterfall plot
    try:
        import IPython
    except ImportError:
        st.error("IPython is not installed. Please install it by adding 'ipython' to your requirements.txt file.")
    else:
        shap.initjs()
  
    fig, ax = plt.subplots(figsize=(15, 8))
    shap.plots.waterfall(shap_values_class[-1], show=False)
    st.pyplot(fig)

# Feature importance plot
st.header("XGBoost ADASYN Model Feature Importance")
st.markdown("""
    The bar chart below shows the feature importance scores for the top 20 features in the XGBoost ADASYN model.
""")

plt.figure(figsize=(10, 8))
feat_importances = pd.Series(xgb_adasyn_model.feature_importances_, index=xc_input.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.title('XGBoost ADASYN Model Feature Importance')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()  # Reverse the order to have the most important feature on top
st.pyplot(plt.gcf())
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd

pickle_folder = 'pickle/'
dataset_folder = 'dataset/'
columns = ['id','name','brand','categories','manufacturer']
class SentimentBasedProductRecommendationSystem:
    def __init__(self):
        self.data = self.read_pickle(pickle_folder + 'processed_data.pkl')
        self.user_final_rating = self.read_pickle(pickle_folder + 'user_final_rating.pkl')
        self.logistic_regression_model =  self.read_pickle(pickle_folder + 'logistic_regression_model.pkl')
        self.raw_data = pd.read_csv(dataset_folder + "sample30.csv")

        self.raw_data['reviews_didPurchase'].fillna(False,inplace=True)
        self.raw_data['reviews_doRecommend'].fillna(False,inplace=True)
        self.raw_data['reviews_title'].fillna('',inplace=True)
        self.raw_data['manufacturer'].fillna('',inplace=True)
        self.raw_data['reviews_username'].fillna('',inplace=True)
        self.raw_data = self.raw_data[self.raw_data['user_sentiment'].notna()]

        self.data = pd.concat([self.raw_data[columns],self.data], axis=1)
        
    def read_pickle(self, file_path):
        return pickle.load(open(file_path,'rb'))

    def recommendProducts(self, user_name):
        items = self.user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index
        features = self.read_pickle(pickle_folder + 'tfidf_vectorizer_features.pkl')
        vectorizer = TfidfVectorizer(vocabulary = features)
        df_prediction=self.data[self.data.id.isin(items)]
        X = vectorizer.fit_transform(df_prediction['review'])
        df_prediction=df_prediction[['id']]
        df_prediction['prediction'] = self.logistic_regression_model.predict(X)
        df_prediction['prediction'] = df_prediction['prediction'].map({'Postive':1,'Negative':0})
        df_prediction=df_prediction.groupby('id').sum()
        df_prediction['positive_reviews']=df_prediction.apply(lambda x: 0.0 if sum(x) == 0 else x['prediction']/sum(x), axis=1)
        product_recommendations=df_prediction.sort_values('positive_reviews', ascending=False).iloc[:5,:].index
        self.data = self.data[self.data.id.isin(product_recommendations)][columns].drop_duplicates()
        self.data = self.data.rename(columns={"id": "Product Id", "name": " Product Name", "brand": "Brand", "categories": "Categories", "manufacturer": "Manufacturer"})
        return self.data.to_html(index=False)

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
        self.model =  self.read_pickle(pickle_folder + 'logistic_reg_model.pkl')
        self.raw_data = pd.read_csv(dataset_folder + "sample30.csv")
        self.data = pd.concat([self.raw_data[columns],self.data], axis=1)
        
    def read_pickle(self, file_path):
        return pickle.load(open(file_path,'rb'))

    def recommendProducts(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        features = self.read_pickle(pickle_folder + 'tfidf_vectorizer_features.pkl')
        vectorizer = TfidfVectorizer(vocabulary = features)
        temp=self.data[self.data.id.isin(items)]
        X = vectorizer.fit_transform(temp['Review'])
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][columns].drop_duplicates().to_html(index=False)
    


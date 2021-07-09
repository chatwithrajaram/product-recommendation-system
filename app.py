from flask import Flask, render_template, request
from model import SentimentBasedProductRecommendationSystem

recommondation_model = SentimentBasedProductRecommendationSystem()
app = Flask(__name__)

file_name = "index.html"

@app.route('/', methods = ['GET'])
def home():
    return render_template(file_name, recommended_products="", show_results=False, show_no_user=False)

@app.route('/list-recommendations', methods = ['POST'])
def fetchRecommendations():
    recommended_products=""
    try:
        recommended_products=recommondation_model.recommendProducts(request.form["user"])
    except:
        recommended_products=""
    return render_template(file_name, recommended_products=recommended_products, show_results=True, show_no_user=True)


if __name__ == '__main__' :
    app.run(debug=True)

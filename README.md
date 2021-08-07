<h1 align='center'>Check-IA ML</h1>
<p align="center">Check-IA API that uses machine learning models to predict your next travel destination :airplane:</p>
<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.7-blue.svg" alt="Python 3.7" />
  </a>
  <img src="https://heroku-badge.herokuapp.com/?app=check-ia-ml" alt="Heroku Status" />
</p>

## ⚒️ Technologies
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [scikit-learn](https://scikit-learn.org/stable/)

## :man_technologist: Machine learning models
- [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

## :computer: Running locally
1. Install the dependencies
```
$ pip install -r requirements.txt
```
2. Run the application
```
$ python app.py
```
3. The application will be running in **localhost:5000**.

## :whale: Running with Docker
1. Run the following commands on terminal
```
$ docker build -t <CONTAINER-NAME>:latest .
$ docker run -p 5000:5000 <CONTAINER-NAME>:latest
```
2. The application will be running in **localhost:5000**.

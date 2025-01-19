import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class Model:
    def __init__(self):
        self.SCALE = 3000000
        dataframe = pd.read_csv("data/Tokenized.csv")
        y = dataframe["FundingAmount"]
        x = dataframe.drop("FundingAmount", axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

        # Rescale
        y_train /= self.SCALE
        y_test /= self.SCALE

        # RANDOM FOREST MODEL
        random_forest = RandomForestRegressor(max_depth=2, random_state=100)
        random_forest.fit(x_train, y_train)

        y_train_prediction = random_forest.predict(x_train)
        y_test_prediction = random_forest.predict(x_test)

        # Evaluate model performance
        train_mean_square_error = mean_squared_error(y_train, y_train_prediction)
        train_r2 = r2_score(y_train, y_train_prediction)
        test_mean_square_error = mean_squared_error(y_test, y_test_prediction)
        test_r2 = r2_score(y_test, y_test_prediction)

        self.model = random_forest
        self.train_mean_square_error = train_mean_square_error
        self.train_r2 = train_r2
        self.test_mean_square_error = test_mean_square_error
        self.test_r2 = test_r2
    def getPrediction(self, state, project_type, reporting_period, disaster_type):
        tokenized_states = {
            "AL": 0,
            "AK": 1,
            "AZ": 2,
            "AR": 3,
            "CA": 4,
            "CO": 5,
            "CT": 6,
            "DC": 7,
            "DE": 8,
            "FL": 9,
            "GA": 10,
            "HI": 11,
            "ID": 12,
            "IL": 13,
            "IN": 14,
            "IA": 15,
            "KS": 16,
            "KY": 17,
            "LA": 18,
            "ME": 19,
            "MD": 20,
            "MA": 21,
            "MI": 22,
            "MN": 23,
            "MS": 24,
            "MO": 25,
            "MT": 26,
            "NE": 27,
            "NV": 28,
            "NH": 29,
            "NJ": 30,
            "NM": 31,
            "NY": 32,
            "NC": 33,
            "ND": 34,
            "OH": 35,
            "OK": 36,
            "OR": 37,
            "PA": 38,
            "RI": 39,
            "SC": 40,
            "SD": 41,
            "TN": 42,
            "TX": 43,
            "UT": 44,
            "VT": 45,
            "VA": 46,
            "WA": 47,
            "WV": 48,
            "WI": 49,
            "WY": 50
        }

        response_types = {
            0: "Develop/Enhance plans procedures and protocols",
            1: "Develop/Enhance interoperable communications systems",
            2: "Develop/Enhance homeland security/emergency management, organization, and structure",
            3: "Develop/enhance state and local geospatial data system/Geographic Information System (GIS)",
            4: "Enhance capabilities to respond to all-hazards events",
            5: "Enhance emergency plans and procedures to reflect the National Response Plan",
            6: "Enhance integration of metropolitan area public health / medical and emergency management capabilities",
            7: "Enhance capability to support economic and community recovery",
            8: "Establish/Enhance emergency plans and procedures to reflect the National Response Plan",
            9: "Establish/Enhance citizen awareness of emergency preparedness prevention and response measures",
            10: "Establish/Enhance emergency operations center",
            11: "Establish/Enhance sustainable Homeland Security exercise program",
            12: "Establish/Enhance citizen/volunteer initiatives",
            13: "Establish / enhance mass care shelter and alternative medical facilities operations",
            14: "Manage, update and/or implement the State Homeland Security Strategy",
            15: "Administer and manage the Homeland Security Grant Program"
        }

        disaster_df = pd.read_csv("data/public_emdat_project.csv")
        index = (disaster_df["Disaster Type"] == disaster_type).idxmax()
        injured_prediction = disaster_df.iloc[index, list(disaster_df.columns).index("No. Injured")]
        deaths_prediction = disaster_df.iloc[index, list(disaster_df.columns).index("Total Deaths")]
        affected_prediction = disaster_df.iloc[index, list(disaster_df.columns).index("Total Affected")]

        state = tokenized_states[state]

        vectorizer = CountVectorizer()
        project = [project_type]
        vectorizer.fit(project)
        vector = vectorizer.transform(project)
        vector_list = vector.toarray().tolist()
        tokenized_project = sum(sum(row) for row in vector_list)

        cost_prediction = (self.model.predict(np.array([state, tokenized_project, reporting_period]).reshape(1, -1)) *
                           self.SCALE)
        cost_prediction = round(cost_prediction[0], 2)
        return [injured_prediction, deaths_prediction, affected_prediction, cost_prediction,
                response_types[tokenized_project]]


    def get_train_MSE(self):
        return self.train_mean_square_error

    def get_train_r2(self):
        return self.train_r2

    def get_test_MSE(self):
        return self.test_mean_square_error

    def get_test_r2(self):
        return self.test_r2

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import csv

response_types = {
    "Develop/Enhance plans procedures and protocols": 0,
    "Develop/Enhance interoperable communications systems": 1,
    "Develop/Enhance homeland security/emergency management organization and structure": 2,
    "Enhance emergency plans and procedures to reflect the National Response Plan": 3,
    "Establish/Enhance emergency plans and procedures to reflect the National Response Plan": 4,
    "Establish/Enhance citizen awareness of emergency preparedness prevention and response measures": 5,
    "Establish/Enhance emergency operations center": 6,
    "Establish/Enhance sustainable homeland security exercise program": 7
}

# disaster_df = pd.read_csv("data/public_emdat_project.csv")
# print(disaster_df.columns)
# disaster_df['No. Injured'] = pd.to_numeric(disaster_df['No. Injured'], errors='coerce')
# index = (disaster_df["Disaster Type"] == "Road").idxmax()
# print(disaster_df.iloc[index, 35])
# result = disaster_df[(disaster_df["Disaster Type"] == "Drought")]
# injured = result.iloc[10]["No. Injured"]
# deaths = result.iloc[0]["Total Deaths"]
# affected = result.iloc[0]["No. Affected"]
# print(injured)

# dataframe = pd.read_csv("data/EmergencyManagementPerformanceGrants.csv")
# col = dataframe["state"]

text = ["the man the play"]

vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
print(vectorizer.vocabulary_)

# encode document
vector = vectorizer.transform(text)

# tokenized list
vector_list = vector.toarray().tolist()
print(sum(sum(row) for row in vector_list))
# tokenized_list = []
# for item in vector_list:
#     tokenized_list.append(item.index(1))

# print(tokenized_list)

# OVERWRITE
# data = {"State": tokenized_list}
# df = pd.DataFrame(data)
# df.to_csv("Tokenized.csv", index=False)

# APPEND
# dataframe2 = pd.read_csv("data/Tokenized.csv")
# dataframe2["State"] = col
# dataframe2.to_csv("Tokenized.csv", index=False)

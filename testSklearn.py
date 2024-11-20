from sklearn.model_selection import train_test_split
import pandas as pd

# Put it into a dataframe
lat = pd.read_table("Latin_stuff/lat_in.trn", sep='\s+', names=['Lemon', 'Infected', 'Infection'])
# I replaced all spaces with hyphens because those entries were messing with read_table
# After reading it in, I replaced the hyphens with spaces
lat = lat.replace('-', ' ', regex=True)

# Separating the data into given and inferred by the program
X = lat[['Lemon', 'Infection']]
y = lat[['Infected']]

# Split data into training and test sets, its a 9:1 split, and 89 just ensures that it's split the same way every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=89)
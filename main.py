"""
מבחן בית לבוטקאמפ למידה עמוקה

מטרת המבחן לבדוק את היכולת שלכם ללמוד חומר חדש לבד. גם אם לא למדתם Machine learning

1) קרא על בעיית סיווג האירוסים של פישר https://en.wikipedia.org/wiki/Iris_flower_data_set
2) השתמש בספריית scikit-learn של פייטון כדי להציג את נתוני הפרחים
3) השתמש בשני אלגוריתמים שונים כדי לסווג את הפרחים לקבוצות הסבר מה למדת
4) הגש את המבחן כפרוייקט ב github שלך
"""


import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor

from tensorflow import keras
from tensorflow.keras import layers


######################
#  הצגת נתוני הפרחים:
######################
iris = load_iris()
# dir(iris)
# ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Class'] = pd.Series(iris.target)

data.head()
"""
   sepal length (cm)  sepal width (cm)  ...  petal width (cm)  Class
0                5.1               3.5  ...               0.2      0
1                4.9               3.0  ...               0.2      0
2                4.7               3.2  ...               0.2      0
3                4.6               3.1  ...               0.2      0
4                5.0               3.6  ...               0.2      0
[5 rows x 5 columns]
"""

data.describe()
"""
       sepal length (cm)  sepal width (cm)  ...  petal width (cm)       Class
count         150.000000        150.000000  ...        150.000000  150.000000
mean            5.843333          3.057333  ...          1.199333    1.000000
std             0.828066          0.435866  ...          0.762238    0.819232
min             4.300000          2.000000  ...          0.100000    0.000000
25%             5.100000          2.800000  ...          0.300000    0.000000
"""


plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

image_path = 'Iris_dataset_scatterplot.png'
image = tf.io.read_file(image_path)
image = tf.io.decode_png(image)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.show()


"""
# print(iris.DESCR)

Iris plants dataset
--------------------
**Data Set Characteristics:**
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica

    :Summary Statistics:
    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================
    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988
The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.
This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.
.. topic:: References
   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
     Mathematical Statistics" (John Wiley, NY, 1950).
   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
     Structure and Classification Rule for Recognition in Partially Exposed
     Environments".  IEEE Transactions on Pattern Analysis and Machine
     Intelligence, Vol. PAMI-2, No. 1, 67-71.
   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
     on Information Theory, May 1972, 431-433.
   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
     conceptual clustering system finds 3 classes in the data.
   - Many, many more ...
"""


######################
# Input Preprocessing
######################
df_train = data.sample(frac=0.7, random_state=0)
df_valid = data.drop(df_train.index)

X_train = df_train.drop('Class', axis=1)
X_valid = df_valid.drop('Class', axis=1)
pre_y_train = df_train['Class']
pre_y_valid = df_valid['Class']

y_train = pd.DataFrame(np_utils.to_categorical(pre_y_train), columns=iris.target_names)
y_train.index = pre_y_train.index
y_valid = pd.DataFrame(np_utils.to_categorical(pre_y_valid), columns=iris.target_names)
y_valid.index = pre_y_valid.index

input_shape = X_train.shape[1]  # = 4
######################


#############################
#   Algorithm 1 - simple NN
#############################

model1 = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[input_shape]),
    layers.Dense(3, activation='softmax')
])
# final_val_accuracy = 96%

model2 = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[input_shape]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')
])
# final_val_accuracy = 93%

model = model1

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    min_delta=0.01,   # Minimum amount of change to count as an improvement.
    patience=10,      # How many epochs to wait before stopping.
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=600,
    verbose=0,
    callbacks=[early_stopping],
)

# Show the learning curves
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
final_val_accuracy = history_df.loc[:, ['val_accuracy']].iloc[-1].iloc[0]
print(f"[Algo 1: NN] final_val_accuracy = {final_val_accuracy}")
# final_val_accuracy = 96%


##################################
#   Algorithm 2 - Decision Tree
##################################
model3 = DecisionTreeRegressor()
model3.fit(X_train, y_train)
predictions = model3.predict(X_valid)
val_accuracy = accuracy_score(y_valid, predictions)
print(f"[Algo 2: DT] val_accuracy = {final_val_accuracy}")
# val_accuracy = 96%

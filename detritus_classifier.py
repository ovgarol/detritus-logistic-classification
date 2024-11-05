import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, confusion_matrix, r2_score
import numpy as np
from itertools import repeat

########################################################################
from scipy.interpolate import UnivariateSpline

# plot metrics as function of training set size
def plot_metrics(train_sizes, metric_list,_label,_color):
    plt.plot(train_sizes, metric_list,color=_color, marker='.',linestyle='',alpha=0.2,label=None, markersize=2)

    # Smoothing the accuracy with a spline
    spline = UnivariateSpline(train_sizes, metric_list, s=.5)  # s is the smoothing factor

    # Generate x values for the spline line
    x_smooth = np.linspace(train_sizes.min(), train_sizes.max(), 500)
    y_smooth = spline(x_smooth)

    # Calculate the confidence interval (using standard error of the mean)
    confidence_interval = 1.96 * np.std(accuracy_list) / np.sqrt(len(accuracy_list))

    # Calculate the upper and lower bounds for the confidence interval
    upper_bound = y_smooth + confidence_interval
    lower_bound = y_smooth - confidence_interval

    # Plotting the smooth spline
    plt.plot(x_smooth, y_smooth, label=_label, color=_color, linestyle='-')

    # Adding the confidence interval
    plt.fill_between(x_smooth, upper_bound, lower_bound, color=_color, alpha=0.2, label=None)


########################################################################
import seaborn as sns

# Plot spectra of interest variables
def plot_spectra(data,name_column,y_pred):
    # Make predictions on the test set
    X_test = data.drop(columns=['ID', 'class','TAXA','DATE','ESD0','ESD','AREA','VOL','PER','CONP','SUMIN'])
    y_pred = pipeline.predict(X_test)

    total_data = data[name_column]
    class_1_data = data[data['class'] == 1][name_column]
    predicted_1_data = data[y_pred == 1][name_column]
    class_0_data = data[data['class'] == 0][name_column]
    predicted_0_data = data[y_pred == 0][name_column]

    # log x axis when required
    is_log = False
    if column_name in ['ESD','AREA','PER','SUMIN','CONP','COMPACT','ELON','VOL','ESD0','ROUGHN','COMPACT']: is_log = True

    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # First subplot
    sns.kdeplot(class_1_data, label='Observed detritus', fill=False, color='black', alpha=0.995, ax=ax1,log_scale=is_log)
    sns.kdeplot(predicted_1_data, label='Predicted detritus', fill=True, color='brown', alpha=0.5, ax=ax1,log_scale=is_log)
    ax1.set_ylabel('Density')
    #ax1.legend()

    # Second subplot
    sns.kdeplot(class_0_data, label='Observed phytoplankton', fill=False, color='black', alpha=0.995, ax=ax2,log_scale=is_log)
    sns.kdeplot(predicted_0_data, label='Predicted phytoplankton', fill=True, color='green', alpha=0.5, ax=ax2,log_scale=is_log)
    ax2.set_ylabel('')
    #ax2.legend()

    # Set the titles and xlabel for the entire figure
    plt.suptitle(name_column+' density distribution ', fontsize=16)

    # Adjust layout
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusts the title position

    # Show or save the plot
    #plt.show()
    plt.savefig(column_name+'.pdf', bbox_inches='tight')  # Save as PDF file with high resolution


########################################################################


########################################################################
# Load the dataset
data = pd.read_csv('data.set')

# Drop the 'ID' column and separate features and target
# Here I ignore size-related variables, e.g., diameter, area, volume, etc.
X = data.drop(columns=['ID', 'class','TAXA','DATE','ESD0','ESD','AREA','VOL','PER','CONP','SUMIN'])
y = data['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)#, random_state=4)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step for feature scaling
    ('logistic', LogisticRegression())  # Logistic Regression model
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall (sensitivity):", recall)
print("Average precision:", avg_precision)

print("Confusion Matrix:\n", conf_matrix)

########################################################################
logistic_model = pipeline.named_steps['logistic']
feature_importance = abs(logistic_model.coef_[0])  # Get coefficients
feature_names = X_train.columns  # Get feature names
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

########################################################################
# Precision-recall curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

display = PrecisionRecallDisplay.from_estimator(
    pipeline, X_test, y_test, name="Logistic Regression"
)
_ = display.ax_.set_title("Precision-Recall curve for detritus classification")

# Calculate the fraction of positives in y
fraction_of_positives = (y == 1).mean()  # '1' is detritus   
plt.axhline(y=fraction_of_positives, color='k', linestyle='--', label='Fraction of Positives')

#plt.show()
plt.savefig('PrecisionRecallCurve.pdf', bbox_inches='tight')  # Save as PDF file with high resolution

########################################################################
# Plots for variable-spectra 
variables_to_plot = data.columns.to_list()

if False: # change to True to plot the distributions
    for column_name in variables_to_plot:
        if column_name in ['ID','DATE','TAXA','class']: continue
        plot_spectra(data,column_name,y_pred)

########################################################################
# Effect of training data set
train_sizes = np.sort(np.random.uniform(0.01, 0.3, 1000))  # 1000 random samples sorted
accuracy_list = []
precision_list = []
recall_list = []
avg_precision_list = []

# Iterate over different training sizes
for train_size in train_sizes:
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)#, random_state=42)

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    avg_precision = average_precision_score(y_test, y_pred)

    # Append metrics to the lists
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    avg_precision_list.append(avg_precision)

# Create a single plot for all metrics
plt.figure(figsize=(6, 4))
plot_metrics(train_sizes, accuracy_list,'Accuracy','blue')
plot_metrics(train_sizes, precision_list, 'Precision', 'orange')
plot_metrics(train_sizes, recall_list, 'Recall', 'green')
plot_metrics(train_sizes, avg_precision_list, 'Average Precision', 'red')

# Adding titles and labels
plt.title('Metrics vs. Training Size')
plt.xlabel('Training Size')
plt.ylabel('Score')
#plt.xscale("log")
#plt.ylim(0, 1)

# Show legend
plt.legend()

# Display the plot
plt.tight_layout()
#plt.show()
plt.savefig('MetricsTrainingSize.pdf', bbox_inches='tight')  # Save as PDF file with high resolution

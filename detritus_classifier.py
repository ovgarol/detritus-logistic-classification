import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, confusion_matrix, r2_score, f1_score
import numpy as np
from itertools import repeat
from scipy import stats

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
    confidence_interval = 1.96 * np.std(accuracy_list) #/ np.sqrt(len(accuracy_list))

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
    #X_test = data.drop(columns=['ID', 'class','TAXA','DATE','ESD0','ESD','AREA','VOL','PER','CONP','SUMIN'])
    #y_pred = pipeline.predict(X_test)
    #data['predicted'] = y_pred

    total_data = data[name_column]
    class_1_data = data[data['class'] == 1][name_column]
    predicted_1_data = data[y_pred == 1][name_column]
    class_0_data = data[data['class'] == 0][name_column]
    predicted_0_data = data[y_pred == 0][name_column]
    w = data[data['class'] == 1]['VOL']

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
    #plt.suptitle(name_column+' density distribution ', fontsize=16)

    # Adjust layout
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusts the title position

    # Show or save the plot
    #plt.show()
    plt.savefig(column_name+'.pdf', bbox_inches='tight')  # Save as PDF file with high resolution


########################################################################

# Plot spectra of interest variables by date
def plot_spectra_by_date(data, name_column, y_pred):
    # Make predictions on the test set
    # X_test = data.drop(columns=['ID', 'class','TAXA','DATE','ESD0','ESD','AREA','VOL','PER','CONP','SUMIN'])
    # y_pred = pipeline.predict(X_test)

    data['predicted'] = y_pred
    total_data = data[name_column]
    class_1_data = data[data['class'] == 1][name_column]
    predicted_1_data = data[y_pred == 1][name_column]
    class_0_data = data[data['class'] == 0][name_column]
    predicted_0_data = data[y_pred == 0][name_column]

    # log x axis when required
    is_log = False
    if column_name in ['ESD','AREA','PER','SUMIN','CONP','COMPACT','ELON','VOL','ESD0','ROUGHN','COMPACT']: is_log = True

    # Create a figure for subplots
    unique_dates = data['DATE'].unique()
    num_dates = len(unique_dates)
    fig, axes = plt.subplots(2, num_dates, figsize=(12, 4), squeeze=False)

    # Initialize lists to store limits
    all_data = []

    # Iterate over each date and plot
    for i, date in enumerate(unique_dates):
        date_data = data[data['DATE'] == date]

        # Separate observed and predicted data
        class_1_data = date_data[date_data['class'] == 1][name_column]
        predicted_1_data = date_data[date_data['predicted'] == 1][name_column]
        class_0_data = date_data[date_data['class'] == 0][name_column]
        predicted_0_data = date_data[date_data['predicted'] == 0][name_column]

        # Store data for global limits
        all_data.extend([class_1_data, predicted_1_data, class_0_data, predicted_0_data])

        # Check if log scale is needed
        is_log = False
        if name_column in ['ESD', 'AREA', 'PER', 'SUMIN', 'CONP', 'COMPACT', 'ELON', 'VOL', 'ESD0', 'ROUGHN', 'COMPACT']:
            is_log = True

        # First subplot for class 1
        sns.kdeplot(class_1_data, label='Observed detritus', fill=False, color='black', alpha=0.995, ax=axes[0,i], log_scale=is_log,linewidth=1, bw_adjust=.5)
        sns.kdeplot(predicted_1_data, label='Predicted detritus', fill=True, color='brown', alpha=0.5, ax=axes[0,i], log_scale=is_log,linewidth=.5, bw_adjust=.5)
        # plt.subplot(2, num_dates,(i+1))
        # if is_log: 
        #     plt.hist(np.log(class_1_data), density=True, bins=30, color='gray')
        #     plt.hist(np.log(predicted_1_data), density=True, bins=30, color='darkgreen',alpha=0.65)

        axes[0,i].set_title(date)
        axes[0,i].set_ylabel('')
        axes[0,i].set_xlabel('')
        #axes[0, i].set_ylabel('density')

        # Perform the Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(class_1_data, predicted_1_data)
        #axes[0,i].text(0.95, 0.95, f'KS Statistic: {ks_statistic:.3f}\nP-value: {p_value:.3f}') 
        # horizontalalignment='right', verticalalignment='top', 
        # transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5)

        # Second subplot for class 0
        sns.kdeplot(class_0_data, label='Observed phytoplankton', fill=False, color='black', alpha=0.995, ax=axes[1,i], log_scale=is_log,linewidth=1, bw_adjust=.5)
        sns.kdeplot(predicted_0_data, label='Predicted phytoplankton', fill=True, color='green', alpha=0.5, ax=axes[1,i], log_scale=is_log,linewidth=.5, bw_adjust=.5)
        axes[1,i].set_ylabel('')
        axes[1,i].set_xlabel('')

        # Perform the Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(class_0_data, predicted_0_data)
        #axes[1,i].text(0.95, 0.95, f'KS Statistic: {ks_statistic:.3f}\nP-value: {p_value:.3f}') 
        # horizontalalignment='right', verticalalignment='top', 
        # transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))


    # Set global limits for x and y axes
    all_data_flat = pd.concat(all_data)
    x_min, x_max = all_data_flat.min(), all_data_flat.max()
    y_min, y_max = 0, max([ax.get_ylim()[1] for ax in axes.flatten()])  # Get max y limit from all axes

    for ax in axes.flatten():
        ax.set_xlim(1.*x_min, 1.*x_max)
        ax.set_ylim(y_min, y_max)

    # fixing x range for size spectra
    if column_name in ['ESD','ESD0']: 
        for ax in axes.flatten(): ax.set_xlim(.8, 200.)
    elif column_name in ['AR','COMPACT','TRANS']: 
        for ax in axes.flatten(): ax.set_xlim(0., 1.)

    # Set labels only for the bottom left subplot
    axes[1, 0].set_xlabel(name_column)
    axes[1, 0].set_ylabel('density')

    # Hide x and y labels for other subplots
    for ax in axes[0, :]:
        ax.set_ylabel('')  # Remove y labels from the top row
        ax.set_xlabel('')  # Remove x labels from the top row

    for ax in axes[1, 1:]:
        ax.set_ylabel('')  # Remove y labels from the bottom row
        ax.set_xlabel('')  # Remove x labels from the right subplots

    # Hide x and y labels for other subplots
    for ax in axes.flatten():
        if ax != axes[1, 0]:  # Only keep labels for the bottom left subplot
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelbottom=False, labelleft=False)  # Hide tick labels

    # Adjust layout
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusts the title position

    # Save the plot as a PDF file
    plt.savefig(f'{name_column}_date.pdf', bbox_inches='tight')
    #plt.show()
########################################################################

def discretize(data, name_column, y_pred):
    name_column = 'SIZECLASS'
    data['predicted'] = y_pred

    # Size discretization
    bins = [0, 2, 5, 10, 20, 50, 100, 200, np.inf]
    names = ['<2', '2-5', '5-10', '10-20', '20-50', '50-100', '100-200','>200']
    data['SIZECLASS'] = pd.cut(data['ESD'], bins, labels=names)

    # Initialize lists to store limits
    all_data = []
    unique_dates = data['DATE'].unique()

    # Iterate over each date and plot
    for i, date in enumerate(unique_dates):
        date_data = data[data['DATE'] == date]

        # Separate observed and predicted data
        total_particles = date_data[name_column] 
        class_1_data = date_data[date_data['class'] == 1][name_column]
        predicted_1_data = date_data[date_data['predicted'] == 1][name_column]
        class_0_data = date_data[date_data['class'] == 0][name_column]
        predicted_0_data = date_data[date_data['predicted'] == 0][name_column]
    
        det_error = 100.0*(class_1_data.value_counts()-predicted_1_data.value_counts())/total_particles.value_counts()#(class_0_data.value_counts()+class_1_data.value_counts())
        det_error = total_particles.value_counts()#(class_0_data.value_counts()+class_1_data.value_counts())
        all_data.append(det_error)
        print(det_error)
        plt.bar(np.array(range(8))+i*8,det_error,color='brown',alpha=0.5)

    plt.axhline(y=0.0, color='k', linestyle='--', label=None,linewidth=0.5)
    #plt.yscale('symlog')
    plt.show()
    plt.clf()
    print('----------------')
    # Convert feature importances to a DataFrame
    all_data = pd.DataFrame(all_data, columns=names)
    all_data.index = unique_dates

    #plt.bar(all_data)
    print(all_data)


########################################################################
########################################################################
if __name__=="__main__":

    # Load the dataset
    data = pd.read_csv('data.all.set')

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
        #('tree', DecisionTreeClassifier())  # Decision Tree Classifier model
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
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall (sensitivity):", recall)
    print("F1-score:", f1)
    print("Average precision:", avg_precision)

    print("Confusion Matrix:\n", conf_matrix)

    ########################################################################
    logistic_model = pipeline.named_steps['logistic']
    feature_importance = (logistic_model.coef_[0])  # Get coefficients
    feature_names = X_train.columns  # Get feature names
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance, 'Abs Importance': abs(feature_importance)})
    importance_df = importance_df.sort_values(by='Abs Importance', ascending=False)
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
    plt.ylim(0.5, 1.0)
    #plt.show()
    plt.savefig('PrecisionRecallCurve.pdf', bbox_inches='tight')  # Save as PDF file with high resolution
    plt.clf()

    ########################################################################
    # Plots for variable-spectra 
    variables_to_plot = data.columns.to_list()
    variables_to_plot = ['ESD','ESD0','AR','ELON','CH1']
    y_pred = pipeline.predict(X)

    if False: # change to True to plot the distributions
        for column_name in variables_to_plot:
            if column_name in ['ID','DATE','TAXA','class']: continue
            plot_spectra(data,column_name,y_pred)

    if False:
        for column_name in variables_to_plot:
            if column_name in ['ID','DATE','TAXA','class']: continue
            plot_spectra_by_date(data,column_name,y_pred)

    discretize(data,'',y_pred)

    ########################################################################
    # Effect of training data set
    if False: 
        train_sizes = np.sort(np.random.uniform(0.01, 0.3, 1000))  # 1000 random samples sorted
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        avg_precision_list = []

        # List to store feature importances for each training size
        feature_importances = []

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
            f1 = f1_score(y_test, y_pred)

            # Append metrics to the lists
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            avg_precision_list.append(avg_precision)
            f1_list.append(f1)

             # Calculate feature importance
            logistic_model = pipeline.named_steps['logistic']
            feature_importance = (logistic_model.coef_[0])  # Get coefficients
            feature_importances.append(feature_importance)  # Store feature importances

        # Convert feature importances to a DataFrame
        importance_df = pd.DataFrame(feature_importances, columns=X_train.columns)

        # Calculate mean and SD of feature importances
        mean_importance = importance_df.mean()
        std_importance = importance_df.std()

        # Create a DataFrame to display mean and SD
        importance_summary = pd.DataFrame({
            'Mean Importance': mean_importance,
            'SD': std_importance,
            'Abs Importance': abs(mean_importance)
        }).sort_values(by='Abs Importance', ascending=False)

        # Print the importance summary
        print(importance_summary)

        from scipy import stats

        # Assuming `importance_summary` DataFrame contains the mean importance
        for feature in importance_summary.index:
            mean_importance = importance_summary.loc[feature, 'Mean Importance']
            std_importance = importance_summary.loc[feature, 'SD']
            
            # Perform a one-sample t-test
            t_stat, p_value = stats.ttest_1samp(importance_df[feature], 0)  # Test against the null hypothesis that the mean is 0
            
            # Print results
            print(f"Feature: {feature}, Mean: {mean_importance:.4f}, p-value: {p_value:.4f}")

        # Create a single plot for all metrics
        plt.figure(figsize=(6, 4))
        plot_metrics(train_sizes, accuracy_list,'Accuracy','blue')
        plot_metrics(train_sizes, precision_list, 'Precision', 'orange')
        plot_metrics(train_sizes, recall_list, 'Recall', 'green')
        plot_metrics(train_sizes, avg_precision_list, 'Average Precision', 'red')
        plot_metrics(train_sizes, f1_list, 'F1 score', 'black')

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

    ########################################################################
    #  mean score metrics and importance metrics
    if True: 
        train_sizes = 0.3+0.*np.sort(np.random.uniform(0.01, 0.3, 1000))  # 1000 random samples sorted
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        avg_precision_list = []

        # List to store feature importances for each training size
        feature_importances = []

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
            f1 = f1_score(y_test, y_pred)

            # Append metrics to the lists
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            avg_precision_list.append(avg_precision)
            f1_list.append(f1)

             # Calculate feature importance
            logistic_model = pipeline.named_steps['logistic']
            feature_importance = (logistic_model.coef_[0])  # Get coefficients
            feature_importances.append(feature_importance)  # Store feature importances

        accuracy_list = np.array(accuracy_list)
        precision_list = np.array(precision_list)
        recall_list = np.array(recall_list)
        f1_list = np.array(f1_list)
        avg_precision_list = np.array(avg_precision_list)

        # Calculate mean and SD of metrics
        print(f"Accuracy: {accuracy_list.mean():.4f}, SD: {accuracy_list.std():.4f}")
        print(f"Precision: {precision_list.mean():.4f}, SD: {precision_list.std():.4f}")
        print(f"Recall: {recall_list.mean():.4f}, SD: {recall_list.std():.4f}")
        print(f"Avg. Precision: {avg_precision_list.mean():.4f}, SD: {avg_precision_list.std():.4f}")
        print(f"F1-score: {f1_list.mean():.4f}, SD: {f1_list.std():.4f}")

        # Convert feature importances to a DataFrame
        importance_df = pd.DataFrame(feature_importances, columns=X_train.columns)

        # Calculate mean and SD of feature importances
        mean_importance = importance_df.mean()
        std_importance = importance_df.std()

        # Create a DataFrame to display mean and SD
        importance_summary = pd.DataFrame({
            'Mean Importance': mean_importance,
            'SD': std_importance,
            'Abs Importance': abs(mean_importance)
        }).sort_values(by='Abs Importance', ascending=False)

        # Print the importance summary
        print(importance_summary, sep='&')

        from scipy import stats

        # Assuming `importance_summary` DataFrame contains the mean importance
        for feature in importance_summary.index:
            mean_importance = importance_summary.loc[feature, 'Mean Importance']
            std_importance = importance_summary.loc[feature, 'SD']
            
            # Perform a one-sample t-test
            t_stat, p_value = stats.ttest_1samp(importance_df[feature], 0)  # Test against the null hypothesis that the mean is 0
            
            # Print results
            print(f"Feature: {feature}, Mean: {mean_importance:.4f}, p-value: {p_value:.4f}")

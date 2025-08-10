#Step 1: Load libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer     #For laonding the inbuilt breast cancer data from sklearn library
from sklearn.preprocessing import StandardScaler    #For standardizing of features to mean=0 and variance=1
from sklearn.decomposition import PCA               #For Principal component Analysis (PCA)
from sklearn.linear_model import LogisticRegression #For Logistic Regression model classification
from sklearn.model_selection import train_test_split #For splitting data into training and test sets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #For evaulating the model's performance
import openpyxl #for writing .xlsx files

#Step 2: Load the breast cancer dataset
data = load_breast_cancer()
#Create dataframe from data and lables its columns
X_raw = pd.DataFrame(data.data, columns=data.feature_names)
#Convert the array 
Y = pd.Series(data.target, name = "target")
features = data.feature_names

#Export raw data to an Excel file
output_path = "C:/Users/HP/Desktop/Raw_Breast_Cancer_Data.xlsx"
X_raw.to_excel(output_path, index=False)
print(f"Raw data successfully saved to: {output_path}")

#Step 3: Standardize the data/
scaler = StandardScaler()   #Create a scaler object
X_scaled = scaler.fit_transform(X_raw)  #Fit data to the scaler and transform it to mean=0 and std=1

#Step 4: Use PCA to identify  essential variables
pca_full = PCA()
pca_full.fit(X_scaled)

loadings_df = pd.DataFrame(
    pca_full.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca_full.components_))],
    index=X_raw.columns
)
loadings_path = "C:/Users/HP/Desktop/PCA_Loadings.xlsx"
loadings_df.to_excel(loadings_path)
print(f"PCA loadings saved to: {loadings_path}")

#Step 5: Apply PCA to reduce components to 2 PCA components
pca = PCA(n_components=2)   #Initialize PCA to reduce data to 2 principal variables
X_pca = pca.fit_transform(X_scaled)   #Fit PCA on the standized data and transform it

#Step 6: Create a dataframe with the principal components
pca_df = pd. DataFrame(data=X_pca, columns=['PC1', 'PC2']) 
pca_df['target'] = Y

#Export the PCA to Excel
output_path = "C:/Users/HP/Desktop/PCA_Results.xlsx"
pca_df.to_excel(output_path, index=False)
print(f"PCA DataFrame succefully saved to: {output_path}")

#Step 7: Split data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)

#Step 8: Implement Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

#Step 9: Make Prediction
Y_pred = logreg.predict(X_test)

#step 10: Evaulate model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, target_names=data.target_names))

#Save prediction results to Excel
results_df = pd.DataFrame({
    'Actual': Y_test.values,
    'Predicted': Y_pred
})
prediction_output_path = "C:/Users/HP/Desktop/PCA_Prediction_Results.xlsx"
results_df.to_excel(prediction_output_path, index=False)
print(f"Prediction results saved to: {prediction_output_path}")

#Step 11: Plot the PCA result
plt.figure(figsize=(8,6))
colors = ['red', 'blue']

#Loop over class lables  and colors to plot each class separately
for lable, color in zip(data.target_names, colors):
    #Select rowws matching the current lable an d plot PC1 and PC2
    plt.scatter(
        pca_df[pca_df['target'] == list(data.target_names).index(lable)]['PC1'],
        pca_df[pca_df['target'] == list(data.target_names).index(lable)]['PC2'],
        label=lable,                #Label for the Lenged
        alpha=0.6,                 #Transparency level
        c=color                     #Colour for the class
    )

#Add plot labels and title
plt.xlabel('Principal Compoment 1')
plt.ylabel('Principal Compoment 2')
plt.title('2-component PCA of Breast Cancer Dataset')   #Title of plot
plt.legend()        #Show legend with class names
plt.grid(True)      #Show gridlines
plt.tight_layout    #Auto-adjustlayout to fit everything
plt.show()          #Display the plot

# Save the plot to desktop before showing
plot_path = "C:/Users/HP/Desktop/PCA_Cancer_Plot.png"
plt.savefig(plot_path, dpi=300) #Save as PNG, 300 DPI
print(f"Plot successfully saved to: {plot_path}")

#Step 12: Explain variance
print("Explained variance by each principal component:")
print(pca.explained_variance_ratio_)        #Print how much variance PC1 and PC2 explain
# PCA-Implementation---Cancer-Center
Overview: The script loads the Breast Cancer dataset, applies Principal Component Analysis (PCA) to reduce the original 30 features to 2 uncorrelated components, trains a Logistic Regression model on the PCA-reduced data, and evaluates the model's performance by predicting target classes on the test data.
Perquisites: Tools used  
  (i)Excel	- for visual review of datasets (raw, transformed, predictions)
  (ii)Pandas - provided functionalities for reading, manipulating and exporting data
  (iii)Numpy -	powers numerical operations and array handling
  (iv)Matplotlib - for creating standard plots
  (v)Sklearn	- supplies data, tools to prepare and analyze the data, and algorithms to build and evaluate predictive model
  (vi)Openpyxl - for reading and writing Excel 
Procedures: 
  (i)Packages including openpyxl, scikit-learn, matplotlib, numpy, pandas have to be installed manually as they are not built-in on Python 
  (ii)The code develops and exports files to Desktop directory; use discretion to edit underlisted code lines to alternative directories (or file names) that may be preferred
      -Line 22: output_path = "C:/Users/HP/Desktop/Raw_Breast_Cancer_Data.xlsx"
      -Line 39: loadings_path = "C:/Users/HP/Desktop/PCA_Loadings.xlsx"
      -Line 52: output_path = "C:/Users/HP/Desktop/PCA_Results.xlsx"
      -Line 79: prediction_output_path = "C:/Users/HP/Desktop/PCA_Prediction_Results.xlsx"
      -Line 109: plot_path = "C:/Users/HP/Desktop/PCA_Cancer_Plot.png"
Code Features:
  (i)Data loading and conversion:
     Load data, converts it to dataframe and save raw dataset to external Excel file
  (ii)PCA operations:
      -Converts raw data to essential components
      -Reduce components to 2 PCA components
      -Save both sets of datasets above to separate external Excel files.


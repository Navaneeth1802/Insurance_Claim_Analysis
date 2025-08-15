<html>
<head>
</head>
<body>
    <p>
        A Machine Learning project to predict the Ultimate Incurred Claim for employees using structured data (numerical + categorical) and unstructured text data (injury descriptions).
    </p>
    <hr>
    <h2> Project Objective</h2>
    <p>
        To build and evaluate regression models capable of accurately predicting the final claim cost of employees' insurance based on demographic, work, and incident-related factors.
    </p>
    <h2>Dataset Overview</h2>
        <table>
            <tr>
                <th>Column Name</th>
                <th>Description</th>
            </tr>
        </thead>
        <table>
        <tbody>
            <tr><td>Year_of_incident</td><td>Year when the incident occurred</td></tr>
            <tr><td>Age</td><td>Age of the employee</td></tr>
            <tr><td>Gender</td><td>Gender of the employee (M, F, etc.)</td></tr>
            <tr><td>MaritalStatus</td><td>Marital status (Single, Married, etc.)</td></tr>
            <tr><td>DependentChildren</td><td>Number of dependent children</td></tr>
            <tr><td>DependentsOther</td><td>Number of Other dependents (parents, spouse, etc.)</td></tr>
            <tr><td>HoursWorkedPerWeek</td><td>Total hours worked in a week</td></tr>
            <tr><td>DaysWorkedPerWeek</td><td>Number of working days per week</td></tr>
            <tr><td>WeeklyWages</td><td>Employee’s weekly wages</td></tr>
            <tr><td>ClaimDescription</td><td>text description of the incident</td></tr>
            <tr><td>InitialIncurredCalimsCost</td><td>Initial estimated claim cost</td></tr>
            <tr><td>UltimateIncurredClaimCost</td><td>Final actual claim cost (Target variable)</td></tr>
        </tbody>
    </table>
    <h2> Preprocessing Workflow</h2>
    <h3> Outlier Handling</h3>
    <ul>
        <li>Applied IQR-based clipping on:
            <ul>
                <li>HoursWorkedPerWeek</li>
                <li>WeeklyWages</li>
                <li>InitialIncurredCalimsCost</li>
                <li>UltimateIncurredClaimCost</li>
            </ul>
        </li>
        <li>Left discrete count features untouched.</li>
    </ul>
    <h3>Skewness Reduction</h3>
    <h4>Applied np.log1p() to reduce skewness in key features like InitialClaim&TargetVariable which had skewness of>1.</h4>
    
  <h3>Text Processing</h3>
  <ul>
      <h4>Used TF-IDF vectorization on column ClaimDescription</h4>
  </ul>

   <h3>Categorical Encoding</h3>
   <ul>
    <h4>One-hot encoded: Gender, MaritalStatus, PartTimeFullTime</li>
    <h3>Feature Scaling</h3>
    <h4>StandardScaler applied to numerical features.</p>
    <h3>Combined Pipeline</h3>
    <h4>Used ColumnTransformer to combine all processing in a clean pipeline.</h4>
  </ul>


<h3>Model Implimentation & Results</h3>

### **1. Baseline Models**

The initial phase involved fitting multiple regression algorithms using **default hyperparameters** to establish baseline performance. The following models were tested:

#### **a. Linear Regression**

* **Implementation:** A simple linear regression model was fit on the transformed training dataset.
* **Objective:** Measure how well a linear relationship captures the patterns in the data without any regularization or parameter tuning.
* **Performance:** Achieved an **R² score of 0.860**, indicating a strong linear correlation between input features and target variable.
* **Error Metrics:**

  * Mean Absolute Error (MAE): **0.323** – average prediction error.
  * Mean Squared Error (MSE): **0.217** – slightly higher penalty for large errors.

#### **b. Decision Tree Regressor**

* **Implementation:** A decision tree was trained with default parameters, meaning no constraints on depth or splitting strategy.
* **Objective:** Capture non-linear relationships by recursively splitting the data based on feature values.
* **Performance:** **R² score of 0.822** – lower than Linear Regression, suggesting overfitting to training data and less generalization on unseen data.
* **Error Metrics:** Higher MAE (**0.366**) and MSE (**0.315**) compared to the linear model, confirming the need for parameter optimization.

#### **c. Gradient Boosting Regressor (GBR)**

* **Implementation:** Used default settings to sequentially train an ensemble of weak learners, where each model corrects errors from the previous one.
* **Objective:** Leverage boosting to improve accuracy and reduce bias.
* **Performance:** Strong results with **R² score of 0.890** – better than both Linear Regression and Decision Tree.
* **Error Metrics:** MAE (**0.276**) and MSE (**0.170**) showed significant improvement, indicating better prediction stability.

#### **d. Random Forest Regressor (RF)**

* **Implementation:** An ensemble of decision trees trained with bootstrap aggregation (bagging) to reduce variance.
* **Objective:** Improve upon single decision tree performance by averaging multiple trees.
* **Performance:** **R² score of 0.887** – competitive with GBR.
* **Error Metrics:** MAE (**0.276**) and MSE (**0.177**) slightly higher than GBR, but still strong.

---

### **2. Hyperparameter Tuning with GridSearchCV**

After establishing baseline metrics, **GridSearchCV** was used for each model to explore a predefined set of hyperparameter combinations. The objective was to identify configurations that maximize performance while avoiding overfitting.

#### **a. Linear Regression with GridSearchCV**

* **Parameters Tested:** `fit_intercept` set to both `True` and `False`.
* **Best Parameters:** `fit_intercept = True`.
* **Performance:** R² improved to **0.874**, confirming that an intercept term slightly boosts performance.
* **Error Metrics:** MAE remained at **0.323**, MSE at **0.217** – no significant change from baseline, meaning linearity still limits accuracy.

#### **b. Decision Tree Regressor with GridSearchCV**

* **Parameters Tested:** Various combinations of `max_depth` (5, 10), `min_samples_split` (2, 5), and split strategies (`best`, `random`).
* **Best Parameters:** `max_depth = 10`, `min_samples_split = 5`, `splitter = best`.
* **Performance:** R² increased to **0.866** – a notable improvement over baseline.
* **Error Metrics:** MAE dropped to **0.327**, MSE to **0.234**, suggesting better generalization.

#### **c. Gradient Boosting Regressor with GridSearchCV**

* **Parameters Tested:** Learning rate (0.01, 0.05, 0.1), number of estimators (100, 150, 200), `max_depth` (3, 5, 10), and `max_features` (`sqrt`, `log2`).
* **Best Parameters:** `learning_rate = 0.05`, `n_estimators = 150`, `max_depth = 10`, `max_features = sqrt`.
* **Performance:** Achieved the highest tuned R² score of **0.909** – confirming strong predictive capability.
* **Error Metrics:** MAE reduced to **0.259**, MSE to **0.155**, the lowest among all models, showing optimal balance between bias and variance.

#### **d. Random Forest Regressor with GridSearchCV**

* **Parameters Tested:** Number of estimators (100, 200), `max_depth` (10, 20), `min_samples_split` (5, 10), `max_features` (`sqrt`, `log2`).
* **Best Parameters:** `max_depth = 20`, `max_features = sqrt`, `min_samples_split = 5`, `n_estimators = 200`.
* **Performance:** R² remained at **0.860** – no significant improvement from baseline, suggesting the model was already near optimal in its default form.
* **Error Metrics:** Matched GBR’s tuned error values (**MAE = 0.259**, **MSE = 0.155**), but without the same boost in R².

---

### **3. Final Comparative Analysis**

* **Best Model:** **Gradient Boosting Regressor (with GridSearchCV)** – highest R² (**0.909**), lowest MAE (**0.259**) and MSE (**0.155**). Demonstrated the best balance of accuracy and error minimization.
* **Worst Model:** **Baseline Decision Tree Regressor** – lowest R² (**0.822**) and highest errors, showing it overfit the training data without tuning.
* **Key Insight:** Ensemble methods (GBR, RF) consistently outperformed standalone models (LR, Decision Tree). Hyperparameter tuning significantly improved Decision Tree and GBR performance, but had minimal effect on Random Forest and Linear Regression.

</body>
</html>

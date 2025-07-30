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

</body>
</html>

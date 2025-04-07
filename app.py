import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from jinja2 import Template
import plotly.express as px
import json
import os
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import skew, kurtosis

app = Flask(__name__)

exclude_cols = [
    # Basic ID variations
    'Id', 'ID', 'id', 'identifier', 'Identifier', 'IDENTIFIER',
    
    # User ID variations
    'userId', 'UserID', 'user_id', 'User_ID', 'UserIdentifier', 'User_Identifier',
    'user_identifier', 'USERID', 'USER_ID', 'UserId',
    'User ID', 'User Id', 'user id', 'User_Id',
    'USER IDENTIFIER', 'User Identifier', 'user identifier',
    
    # Customer ID variations
    'customer_id', 'CustomerID', 'CustomerId', 'CUSTOMER_ID', 'Customer_ID',
    'Customer ID', 'Customer Id', 'customer id', 'Customer_Id',
    'CustomerIdentifier', 'Customer_Identifier', 'customer_identifier',
    
    # Contact Information
    ## Phone/Mobile variations
    'mobile', 'Mobile', 'MOBILE', 'mobile_number', 'Mobile_Number',
    'phone', 'Phone', 'PHONE', 'PhoneNumber', 'phone_number', 'Phone_Number',
    'telephone', 'Telephone', 'tel', 'Tel', 'TEL',
    'contact_number', 'ContactNumber', 'Contact_Number',
    'cell', 'Cell', 'CellPhone', 'cell_phone',
    'mobile_phone', 'MobilePhone', 'Mobile_Phone',
    
    ## Number variations
    'number', 'Number', 'NUMBER', 'num', 'Num', 'NUM',
    
    ## Email variations
    'email', 'Email', 'EMAIL', 'EmailAddress', 'email_address', 'Email_Address',
    'mail', 'Mail', 'MAIL', 'e_mail', 'E_Mail', 'E-mail',
    'EmailID', 'email_id', 'Email_ID',
    
    # Account related
    'account_number', 'AccountNumber', 'Account_Number', 'ACCOUNT_NUMBER',
    'AccountID', 'account_id', 'Account_ID', 'ACCOUNT_ID',
    'AcctNum', 'acct_num', 'Acct_Num',
    'account', 'Account', 'ACCOUNT',
    
    # Government IDs
    ## SSN variations
    'ssn', 'SSN', 'SocialSecurityNumber', 'social_security_number',
    'Social_Security_Number', 'SOCIAL_SECURITY_NUMBER',
    'ssn_number', 'SSN_Number', 'SSNNumber',
    
    ## Passport variations
    'passport', 'Passport', 'PASSPORT',
    'PassportNumber', 'passport_number', 'Passport_Number',
    'passport_id', 'PassportID', 'Passport_ID',
    
    ## License variations
    'license', 'License', 'LICENSE',
    'LicenseNumber', 'license_number', 'License_Number',
    'driver_license', 'DriversLicense', 'Drivers_License',
    'driving_license', 'DrivingLicense', 'Driving_License',
    'dl_number', 'DL_Number', 'DLNumber',
    
    # Location Information
    ## Address variations
    'address', 'Address', 'ADDRESS',
    'street_address', 'StreetAddress', 'Street_Address',
    'mailing_address', 'MailingAddress', 'Mailing_Address',
    'residence', 'Residence', 'RESIDENCE',
    'location', 'Location', 'LOCATION',
    
    ## Postal Code variations
    'postal_code', 'PostalCode', 'Postal_Code', 'POSTAL_CODE',
    'zip', 'Zip', 'ZIP',
    'zipcode', 'ZipCode', 'Zip_Code', 'ZIP_CODE',
    'pin', 'Pin', 'PIN',
    'pincode', 'PinCode', 'Pin_Code', 'PIN_CODE',
    
    # Additional Personal Identifiers
    'national_id', 'NationalID', 'National_ID',
    'tax_id', 'TaxID', 'Tax_ID',
    'passport_no', 'PassportNo', 'Passport_No',
    'aadhar', 'Aadhar', 'AADHAR',
    'pan', 'PAN', 'Pan',
    
    # Composite variations
    'user_contact', 'UserContact', 'User_Contact',
    'customer_contact', 'CustomerContact', 'Customer_Contact',
    'personal_id', 'PersonalID', 'Personal_ID',
    'unique_id', 'UniqueID', 'Unique_ID'
]

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def analyze_dataset(df):
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    non_numeric_cols = df.select_dtypes(exclude=[np.number])

    # Impute missing values for numeric columns with mean
    if not numeric_cols.empty:
        numeric_cols = numeric_cols.fillna(numeric_cols.mean())

    # Display target variable type
    target = df.iloc[:, -1]  # Assumes the last column is the target column
    target_type = 'Categorical' if target.dtype == 'object' else 'Numerical'

    # Identify missing values
    missing_values = df.isnull().sum().to_dict()

    # Identify columns with potential outliers (numeric only)
    outlier_columns = []
    for col in numeric_cols.columns:
        lower_limit, upper_limit = np.percentile(numeric_cols[col], [5, 95])
        if ((numeric_cols[col] < lower_limit).sum() > 0) or ((numeric_cols[col] > upper_limit).sum() > 0):
            outlier_columns.append(col)

    # Calculate skewness and kurtosis for numeric columns
    skewness = numeric_cols.skew().to_dict() if not numeric_cols.empty else {}
    kurtosis_values = numeric_cols.kurtosis().to_dict() if not numeric_cols.empty else {}

    # Check for multicollinearity (numeric only)
    high_corr_pairs = []
    if len(numeric_cols.columns) > 1:
        corr_matrix = numeric_cols.corr()
        high_corr_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j]) 
                           for i in range(len(corr_matrix.columns)) 
                           for j in range(i) 
                           if abs(corr_matrix.iloc[i, j]) > 0.8]

    # Check for linear relationship with the target (only if the target is numerical)
    linear_features = []
    if target_type == 'Numerical':
        f_scores, _ = f_regression(numeric_cols, target)
        linear_features = numeric_cols.columns[f_scores > np.percentile(f_scores, 75)].tolist()

    # Prepare recommendations
    recommendations = []
    if target_type == 'Categorical':
        recommendations.append("Target variable is categorical. Consider using classification models.")
    else:
        recommendations.append("Target variable is numerical. Consider using regression models.")

    if sum(missing_values.values()) > 0:
        recommendations.append("Missing values present. Consider imputation strategies.")
        if sum(missing_values.values()) < len(df) * 0.1:
            recommendations.append("Number of missing values is relatively small. Consider Median Imputation.")
        else:
            recommendations.append("Use Mean Imputation for numeric features or Mode Imputation for categorical features.")

    if outlier_columns:
        recommendations.append(f"Outliers detected in columns: {', '.join(outlier_columns)}. Consider outlier treatment techniques.")

    if not numeric_cols.empty and max(abs(v) for v in skewness.values()) > 0.5:
        recommendations.append("Some numeric features are skewed. Consider transformations like log, Box-Cox, or square root.")

    if not numeric_cols.empty and numeric_cols.apply(np.std).max() > 1:
        recommendations.append("Features have varying scales. Consider normalizing or standardizing the data.")

    if high_corr_pairs:
        recommendations.append("Multicollinearity detected. Consider regularization techniques or feature selection methods.")

    # Return a dictionary summarizing the analysis
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'target_type': target_type,
        'missing_values': missing_values,
        'outlier_columns': outlier_columns,
        'skewness': skewness,
        'kurtosis': kurtosis_values,
        'high_corr_pairs': high_corr_pairs,
        'linear_features': linear_features,
        'recommendations': recommendations
    }

def preprocess_data(df, target_col):
    # Define columns to keep (excluding target column and excluded columns)
    X = df.drop(columns=[target_col] + [col for col in df.columns if any(excl in col for excl in exclude_cols)], axis=1)
    y = df[target_col]

    target_is_numerical = y.dtype != 'object'
    preprocessing_steps = []

    # Check numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Check for missing values and recommend appropriate imputation
    for col in numerical_cols:
        if X[col].isnull().any():
            # Calculate skewness for the column
            skewness = X[col].skew()
            # If data is significantly skewed (abs(skewness) > 0.5), recommend median imputation
            if abs(skewness) > 0.5:
                preprocessing_steps.append(f"Impute missing values in '{col}' using median imputation (numerical variable with skewed distribution)")
            else:
                preprocessing_steps.append(f"Impute missing values in '{col}' using mean imputation (numerical variable with normal distribution)")
    
    for col in categorical_cols:
        if X[col].isnull().any():
            preprocessing_steps.append(f"Impute missing values in '{col}' using mode imputation (categorical variable)")

    # Check for outliers in numerical columns
    for col in numerical_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        if ((X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))).any():
            preprocessing_steps.append(f"Winsorize outliers in '{col}' to reduce their influence on the model")

    # Recommend standardization for numerical features if needed
    if len(numerical_cols) > 0:
        if (X[numerical_cols].std() > 10).any():
            preprocessing_steps.append("Standardize numeric features to ensure all features are on the same scale")

    # Recommend encoding for categorical features
    if len(categorical_cols) > 0:
        preprocessing_steps.append("Label encode categorical features for model compatibility")

    # Handle target variable
    if not target_is_numerical:
        preprocessing_steps.append("Encode categorical target variable into numerical values for classification models")

    return X, y, preprocessing_steps

def recommend_model(df, target_col, dataset_characteristics):
    target_type = df[target_col].dtype
    is_classification = target_type == 'object' or target_type == 'str'
    
    if is_classification:
        models = [
            (LogisticRegression(), "Logistic Regression", "Simple, interpretable model for binary classification"),
            (KNeighborsClassifier(), "k-Nearest Neighbors", "Non-linear model, good for small to medium datasets"),
            (DecisionTreeClassifier(), "Decision Tree", "Non-linear model, handles both categorical and continuous features"),
            (RandomForestClassifier(), "Random Forest", "Ensemble method, good for complex relationships"),
            (SVC(), "Support Vector Classifier", "Effective for non-linear boundaries, works well with scaled data"),
            (GaussianNB(), "Naive Bayes", "Fast, works well with high-dimensional data"),
            (GradientBoostingClassifier(), "Gradient Boosting Classifier", "Powerful ensemble method for complex datasets")
        ]
    else:
        models = [
            (LinearRegression(), "Linear Regression", "Simple, interpretable model for linear relationships"),
            (Ridge(), "Ridge Regression", "Linear model with L2 regularization, good for multicollinearity"),
            (Lasso(), "Lasso Regression", "Linear model with L1 regularization, good for feature selection"),
            (ElasticNet(), "Elastic Net", "Combines L1 and L2 regularization"),
            (SVR(), "Support Vector Regression", "Effective for non-linear relationships"),
            (DecisionTreeRegressor(), "Decision Tree Regressor", "Non-linear model, handles both categorical and continuous features"),
            (RandomForestRegressor(), "Random Forest Regressor", "Ensemble method, good for complex relationships"),
            (GradientBoostingRegressor(), "Gradient Boosting Regressor", "Powerful ensemble method for complex datasets")
        ]

    recommended_models = []
    for model, name, description in models:
        score = 0
        
        # General conditions
        if dataset_characteristics['n_samples'] > 10000:
            if isinstance(model, (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor)):
                score += 2
        else:
            if isinstance(model, (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SVC, SVR)):
                score += 1

        if dataset_characteristics['n_features'] > 100:
            if isinstance(model, (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor)):
                score += 2
        
        # Specific conditions for regression models
        if not is_classification:
            if dataset_characteristics.get('linear_relationship', False):
                if isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet)):
                    score += 3
            else:
                if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor, SVR)):
                    score += 2
            
            if dataset_characteristics.get('multicollinearity', False):
                if isinstance(model, (Ridge, Lasso, ElasticNet)):
                    score += 2
            
            if dataset_characteristics.get('outliers', False):
                if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor, SVR)):
                    score += 2
                elif isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet)):
                    score -= 1
        
        # Specific conditions for classification models
        else:
            if dataset_characteristics.get('linear_relationship', False):
                if isinstance(model, (LogisticRegression)):
                    score += 3
            else:
                if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, SVC)):
                    score += 2
            
            if dataset_characteristics.get('multicollinearity', False):
                if isinstance(model, (LogisticRegression)):
                    score -= 1
                elif isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                    score += 1
            
            if dataset_characteristics.get('outliers', False):
                if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, SVC)):
                    score += 2
                elif isinstance(model, (LogisticRegression)):
                    score -= 1
        
        # Additional conditions
        if dataset_characteristics.get('categorical_features', False):
            if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor)):
                score += 2
        
        if dataset_characteristics.get('high_dimensionality', False):
            if isinstance(model, (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor)):
                score += 2
        
        recommended_models.append((model, name, description, score))

    recommended_models.sort(key=lambda x: x[3], reverse=True)
    
    return recommended_models[:4]  # Return top 4 recommended models

def generate_visualizations(df):
    visualizations = []

    # Filter out columns to exclude from visualization
    columns_to_visualize = [col for col in df.columns if not any(excl.lower() in col.lower() for excl in exclude_cols)]

    # Pair plot for numerical columns
    numerical_cols = df[columns_to_visualize].select_dtypes(include=[np.number]).columns
    # if len(numerical_cols) > 1:
    #     fig = px.scatter_matrix(df[numerical_cols])
    #     visualizations.append({
    #         'name': 'Pair Plot',
    #         'plot': json.loads(fig.to_json())
    #     })

    # Correlation heatmap
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        fig.update_layout(title="Correlation Heatmap")
        visualizations.append({
            'name': 'Correlation Heatmap',
            'plot': json.loads(fig.to_json())
        })

    # Distribution plots for numerical columns
    for col in numerical_cols[:5]:  # Limit to first 5 numerical columns
        fig = px.histogram(df, x=col, marginal="box")
        fig.update_layout(title=f"Distribution of {col}")
        visualizations.append({
            'name': f'Distribution of {col}',
            'plot': json.loads(fig.to_json())
        })

    # Bar plot for categorical columns
    categorical_cols = df[columns_to_visualize].select_dtypes(include=['object']).columns
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        value_counts = df[col].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values)
        fig.update_layout(title=f"Value Counts of {col}")
        visualizations.append({
            'name': f'Value Counts of {col}',
            'plot': json.loads(fig.to_json())
        })

    return visualizations

def generate_report_html(analysis_data, preprocessing_data):
    # HTML template for the report
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .section { margin-bottom: 30px; }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; margin-top: 20px; }
            .dashboard-card { 
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .recommendation-item {
                background: #e8f4f8;
                padding: 10px;
                margin: 5px 0;
                border-radius: 3px;
            }
            .visualization-card {
                margin: 20px 0;
                padding: 10px;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .model-card {
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .progress-bar {
                background: #eee;
                height: 20px;
                border-radius: 10px;
                overflow: hidden;
            }
            .progress {
                background: #4CAF50;
                height: 100%;
            }
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Data Analysis Report</h1>
        
        <div class="section">
            <h2>Dataset Information</h2>
            <div class="dashboard-card">
                <p>Number of Rows: {{ analysis_data.shape[0] }}</p>
                <p>Number of Columns: {{ analysis_data.shape[1] }}</p>
                <p>Target Type: {{ analysis_data.target_type }}</p>
            </div>
        </div>

        <div class="section">
            <h2>Missing Values</h2>
            <div class="dashboard-card">
                {% for column, count in analysis_data.missing_values.items() %}
                <p>{{ column }}: {{ count }}</p>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>Outlier Columns</h2>
            <div class="dashboard-card">
                {% for column in analysis_data.outlier_columns %}
                <p>{{ column }}</p>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>High Correlation Pairs</h2>
            <div class="dashboard-card">
                {% for pair in analysis_data.high_corr_pairs %}
                <p>{{ pair[0] }} - {{ pair[1] }}</p>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>Analysis Recommendations</h2>
            {% for rec in analysis_data.recommendations %}
            <div class="recommendation-item">{{ rec }}</div>
            {% endfor %}
        </div>

        <div class="section">
            <h2>Visualizations</h2>
            {% for viz in visualizations %}
            <div class="visualization-card">
                <h3>{{ viz.name }}</h3>
                <div id="{{ viz.name | replace(' ', '-') }}"></div>
                <script>
                    Plotly.newPlot("{{ viz.name | replace(' ', '-') }}", 
                                 {{ viz.plot.data | tojson }}, 
                                 {{ viz.plot.layout | tojson }});
                </script>
            </div>
            {% endfor %}
        </div>

        {% if preprocessing_data %}
        <div class="section">
            <h2>Preprocessing Steps</h2>
            <div class="dashboard-card">
                {% for step in preprocessing_data.preprocessing_steps %}
                <p>{{ step }}</p>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>Recommended Models</h2>
            {% for model in preprocessing_data.recommended_models %}
            <div class="model-card">
                <h3>{{ model.name }}</h3>
                <p>{{ model.description }}</p>
                <div class="progress-bar">
                    <div class="progress" style="width: {{ (model.score / 3) * 100 }}%"></div>
                </div>
                <p>Score: {{ "%.2f"|format(model.score) }}</p>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </body>
    </html>
    """
    
    template = Template(template_str)
    return template.render(
        analysis_data=analysis_data,
        preprocessing_data=preprocessing_data,
        visualizations=analysis_data.get('visualizations', [])
    )
    
@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        analysis_data = data.get('analysis_data')
        preprocessing_data = data.get('preprocessing_data')
        
        if not analysis_data:
            return jsonify({'error': 'No analysis data provided'}), 400
            
        report_html = generate_report_html(analysis_data, preprocessing_data)
        
        return jsonify({
            'report_html': report_html
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            df = pd.read_csv(filename)
            analysis = analyze_dataset(df)
            visualizations = generate_visualizations(df)
            return jsonify({**analysis, 'visualizations': visualizations})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return jsonify({'error': 'Unknown error occurred'}), 400

@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.json
    filename = os.path.join(UPLOAD_FOLDER, data['filename'])
    target_col = data['target_col']
    
    try:
        df = pd.read_csv(filename)
        X, y, preprocessing_steps = preprocess_data(df, target_col)
        dataset_characteristics = {
            'target_type': 'Categorical' if df[target_col].dtype == 'object' else 'Numerical',
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'linear_relationship': any(f in df.columns for f in data.get('linear_features', [])),
            'multicollinearity': len(data.get('high_corr_pairs', [])) > 0,
            'outliers': len(data.get('outlier_columns', [])) > 0,
            'categorical_features': len(df.select_dtypes(include=['object']).columns) > 0,
            'high_dimensionality': X.shape[1] > 100
        }
        recommended_models = recommend_model(df, target_col, dataset_characteristics)
        return jsonify({
            'preprocessing_steps': preprocessing_steps,
            'recommended_models': [
                {'name': name, 'description': desc, 'score': score} 
                for _, name, desc, score in recommended_models
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

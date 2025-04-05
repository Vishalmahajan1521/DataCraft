# DataCraft: Your AI-Powered Data Analysis Sidekick 🚀

Tired of wrestling with messy data? 😩 **DataCraft** is a Python-based web application designed to automate and streamline your data science workflow, empowering both novice and experienced data scientists to gain insights quickly and efficiently! ✨

## What is DataCraft? 🤔

DataCraft is your intelligent assistant 🤖 that simplifies Exploratory Data Analysis (EDA) and delivers actionable insights for your datasets. Upload your data, and let DataCraft do the heavy lifting! 💪

### Key Features 🌟

- **Automated EDA** 🤖: Instantly performs comprehensive Exploratory Data Analysis, delivering a detailed overview in seconds! ⏱️
- **Data Quality Detective** 🔍: Identifies common data issues, including:
  - **Missing Values** 🕳️: Detects gaps in your data.
  - **Outliers** ⚠️: Flags extreme values that may skew results.
  - **Skewness** 📉: Measures asymmetry in data distributions.
  - **Multicollinearity** 📈: Highlights highly correlated features.
- **Actionable Recommendations** 💡: Provides tailored suggestions for:
  - **Preprocessing Techniques** 🛠️: Imputation, scaling, and encoding strategies.
  - **Machine Learning Models** 🧠: Recommends optimal regression or classification models.
- **Visualizations** 📊: Generates insightful plots (e.g., correlation heatmaps, histograms) using Plotly.

### Why DataCraft? ❓

- **Saves Time** ⏰: Automates repetitive tasks, freeing you for deeper analysis.
- **Improves Accuracy** 💯: Offers objective, statistically grounded recommendations.
- **Enhances Collaboration** 🤝: Produces shareable HTML reports for seamless teamwork.

## Tech Stack 🛠️

- **Python** 🐍: Core programming language.
- **Flask** 🌐: Web framework for the application.
- **Pandas** 🐼: Data manipulation and analysis.
- **NumPy** 🔢: Numerical computations.
- **Scikit-learn** 🤖: Machine learning models and preprocessing.
- **Plotly** 📈: Interactive visualizations.
- **SciPy** 🔬: Scientific computing and statistical analysis.
- **Jinja2** 🖌️: Templating for HTML reports.

## Installation ⚙️

1. **Clone the Repository** 📥:
   ```bash
   git clone https://github.com/Vishalmahajan1521/DataCraft.git
   cd DataCraft
2. **Install Dependencies** 📦:
   ```bash
   pip install -r requirements.txt
3. **Run the Application** 🚀:
   ```bash
   python app.py
  
Open your browser and navigate to http://127.0.0.1:5000 🌐.

## Usage 🎯
**Upload a Dataset** 📤: Use the web interface to upload a CSV file.
**Analyze Data** 🔍: DataCraft performs EDA and displays results like missing values, outliers, and recommendations.
**Preprocess and Model** 🛠️: Specify a target column for preprocessing steps and model suggestions.
**Generate Reports** 📜: Export analysis results as an HTML report.

## Project Structure 📂
```bash
DataCraft/
├── app.py              # Main Flask application 🌐
├── uploads/            # Directory for uploaded files 📁
├── templates/          # HTML templates (index.html) 📝
└── requirements.txt    # Dependencies 📋


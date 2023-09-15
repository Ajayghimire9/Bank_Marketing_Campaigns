# Bank_Marketing_Campaigns
# Portugal Bank Marketing Campaigns

## Abstract

This project analyzes the results of bank marketing campaigns conducted by a Portuguese bank. The campaigns were mainly based on direct phone calls, where the bank offered its clients the opportunity to place a term deposit. The dataset captures various attributes such as age, job type, marital status, etc., and the outcome (yes/no) indicating whether the client subscribed to a term deposit.

## Tasks

- **Classification**: Predict the future results of marketing campaigns based on available statistics and formulate recommendations for future campaigns.
- **Consumer Profiling**: Build a profile of a typical consumer for the bank's services, particularly term deposits.
- **Clustering**: Use KNN clustering to create imaginary boundaries for data classification.

## Features Description

### Bank Client Data
- **Age**: Numeric
- **Job**: Type of job (admin, blue-collar, entrepreneur, etc.)
- **Marital Status**: (divorced, married, single, unknown)
- **Education**: Level of education (basic, high school, professional course, etc.)
- **Default**: Has credit in default? (yes, no, unknown)
- **Housing**: Has housing loan? (yes, no, unknown)
- **Loan**: Has personal loan? (yes, no, unknown)

### Last Contact Data
- **Contact**: Type of communication (cellular, telephone)
- **Month**: Last contact month
- **Day of Week**: Last contact day of the week
- **Duration**: Last contact duration in seconds

### Other Attributes
- **Campaign**: Number of contacts during this campaign
- **PDays**: Number of days passed since the client was last contacted from a previous campaign
- **Previous**: Number of contacts before this campaign
- **POutcome**: Outcome of the previous marketing campaign

### Social and Economic Context Attributes
- **Employment Variation Rate**: Quarterly indicator
- **Consumer Price Index**: Monthly indicator
- **Consumer Confidence Index**: Monthly indicator
- **Euribor 3-Month Rate**: Daily indicator
- **Number of Employees**: Quarterly indicator

### Target Variable
- **Y**: Has the client subscribed to a term deposit? (yes, no)

## Source Code

The analysis is conducted using Python, leveraging libraries such as Pandas, NumPy, Matplotlib, and scikit-learn. The complete code can be found in the repository.

## Results

The models used for this analysis include Logistic Regression, Random Forest, KNN, Decision Trees, and Bagging Classifiers. The best-performing model is the Random Forest Classifier with an accuracy of approximately 91%.

## Insights
- The most influential features in determining whether a client will subscribe to a term deposit are 'duration of the call', 'age', and 'employment variation rate'.
- The Random Forest Classifier performed the best among all models, with an accuracy of 91%.

## Conclusion

This analysis provides insights into the effectiveness of bank marketing campaigns and offers recommendations for future campaigns. Further improvement can be achieved by fine-tuning the model and possibly by incorporating more features.

## Getting Started

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run `Bank_Marketing_Campaigns.py` to perform the analysis.

## Acknowledgements

- Dataset Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

For more details, please refer to the Jupyter Notebook in the repository.

Feel free to contribute!

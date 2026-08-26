# SCT_DS_3 — Bank Marketing Decision Tree Classifier

## SkillCraft Technology — Data Science Internship — Task 3

### Task Objective
Build a Decision Tree Classifier to predict whether a bank customer will subscribe to a product/service based on demographic and behavioral information.

### Dataset
This project uses the Bank Marketing dataset. The uploaded CSV contains customer information such as:

- Age
- Job
- Marital status
- Education
- Default status
- Account balance
- Housing loan
- Personal loan
- Contact method
- Contact day/month
- Call duration
- Campaign information
- Previous contact information
- Previous campaign outcome

The target column is `y`:
- `yes` → customer subscribed
- `no` → customer did not subscribe

### Project Workflow
1. Load and inspect the dataset.
2. Remove duplicate rows.
3. Separate features and target.
4. One-hot encode categorical variables.
5. Split the data into training and testing sets using stratification.
6. Train a Decision Tree Classifier.
7. Evaluate the model using accuracy, precision, recall and F1-score.
8. Generate a confusion matrix.
9. Visualize the decision tree.
10. Calculate permutation-based feature importance.

### Model
`DecisionTreeClassifier` is used with controlled tree depth and minimum leaf/split sizes to reduce overfitting.

### How to Run

```bash
pip install -r requirements.txt
python decision_tree_classifier.py
```

The script generates:
- `visualizations/confusion_matrix.png`
- `visualizations/decision_tree.png`
- `visualizations/feature_importance.png`
- `visualizations/feature_importance.csv`

### Important Note
The `duration` feature describes the duration of the marketing contact. It can be highly predictive, but in a real deployment it may not be available before a call takes place. Therefore, it should be excluded if the goal is to make predictions before contacting the customer.

### Repository Naming
This repository follows the internship instruction:
`SCT_TrackCode_TaskNumber`

For the Data Science track:
`SCT_DS_3`

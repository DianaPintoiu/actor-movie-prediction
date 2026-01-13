# actor-movie-prediction
This project focuses on developing a solution for the prediction of what next movies will actors have based on their earlier filmography. 

## Colab link:
https://colab.research.google.com/drive/1c_zHgq9do8AEGl8qTREWc3LfVwBoncND#scrollTo=f2914f76

## Good to know articles

1. **A Statistical Analysis of Gross Revenue in Movie Industry(Chen, 2011)** -> https://www.ijbmer.com/docs/volumes/vol9issue3/ijbmer2018090303.pdf

   - relates to gross revenue to budget and other features.
   - is a statistical regression model
   - paper talks about how they ran **linear regressions** where gross revenue (dependent variable) depends on production budget, genre dummies, runtime, etc.
   - Result: they found that a **quadratic term in budget** improved the model; this suggests **non-linear returns** of increasing budget.  



## Movie Success Classifier (Budget vs. Box Office)

We classified a film as **Loss**, **Break-even**, or **Hit** using ROI metrics.

### 1. ROI Metric

**ROI Ratio**
ROI_ratio = Gross / Budget

**ROI Percentage**
ROI_% = (Gross - Budget) / Budget * 100

### 2. Rule-Based Classifier

- **Loss:** ROI_ratio < 1.0  
- **Break-even:** 1.0 ≤ ROI_ratio < 2.5  
- **Hit:** ROI_ratio ≥ 2.5  


### 3. Multinomial Logistic Regression

Predicts probabilities for the three classes:
P(Y = k | X) = exp(alpha_k + beta_k^T X) / sum_m exp(alpha_m + beta_m^T X)

- `X` = predictors (ROI_ratio or log_ROI, genre dummies, sequel, etc.)  
- `alpha_k` = intercept for class k  
- `beta_k` = coefficients for class k  
- Predicted class: `Y_hat = argmax_k P(Y=k|X)`

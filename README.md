# 📌 FlipItNews: AI-Powered News Categorization  

## 🚀 Problem Statement  
FlipItNews aims to **simplify finance, business, and investment news for millennials** using AI/ML-driven content discovery. Our goal is to **categorize news articles** into **Politics, Technology, Sports, Business, and Entertainment** using NLP-based multi-class classification.  

---

## 🔍 Dataset Insights  
- **Total Articles:** 2225  
- **Balanced Distribution Across Categories:**  
  - 🏆 **Sports:** 511  
  - 📈 **Business:** 510  
  - 🗳 **Politics:** 417  
  - 📡 **Technology:** 401  
  - 🎭 **Entertainment:** 386  

---

## 🛠 Methodology  

### 1️⃣ Text Preprocessing with `TextPreprocessor`  
✔ **Cleaning**: Lowercasing, removing HTML tags, URLs, mentions, hashtags, numbers, emails, and extra spaces. expanding contractions
✔ **Tokenization**: Splitting text into individual words.  
✔ **Word Segmentation**: Handling concatenated words (*e.g.,* `"lookabout"` → `"look about"`).  
✔ **Spell Correction**: Fixing misspelled words (excluding proper nouns).  
✔ **Lemmatization**: Converting words to their base form.  
✔ **Rare/Common Word Removal**: Filtering out words occurring **>1000 times** or **too rarely**.  

---

### 2️⃣ Feature Engineering  
- **Extracted Noun Phrases** for key insights per category.  
- **Generated WordClouds** to visualize dominant terms in each category.  

#### Example Top Phrases by Category:  
- **📈 Business:** *market, bank, share, economy, price*  
- **🎭 Entertainment:** *award, music, show, actor, song*  
- **🗳 Politics:** *party, election, minister, plan, labor*  
- **🏆 Sports:** *player, win, match, team, side*  
- **📡 Technology:** *technology, mobile, phone, service, user*  

---

## 🏆 Model Training & Evaluation  

### ✅ Model Selection & Hyperparameter Tuning  
I implemented **Stratified K-Fold Cross-Validation (5-Fold)** to ensure a balanced evaluation across all classes. Hyperparameter tuning was conducted using a **custom grid search approach**.  

#### **Explored Models**:  
- 🏛 **Naïve Bayes** (Baseline Model)  
- 🔍 **K-Nearest Neighbors (KNN)**  
- 🌲 **Random Forest**  
- 🚀 **XGBoost Classifier** (Best Model)  
- 🏗 **Decision Tree**  

#### **Best Performing Model:**  
🏆 **XGBoostClassifier** achieved **95% accuracy & 97% NDCG (Top 2 relevant ranks).**  

---

## 🧐 Model Performance  

| Model | Validation Accuracy | Validation NDCG (Top 2) | Overfitting? |
|--------|-------------------|-------------------|--------------|
| **GaussianNB** | 85% | 87% | ✅ Yes (100% Train Accuracy) |
| **KNeighborsClassifier** (`metric='cosine', n_neighbors=7`) | 92% | 96% | ❌ No |
| **RandomForestClassifier** (`max_depth=5, n_estimators=200, oob_score=True`) | 90% | 93% | ❌ No |
| **XGBoostClassifier** | **95%** | **97%** | ❌ No |

---

## 🔎 Model Interpretability  
- **LIME (Local Interpretable Model-Agnostic Explanations)** was used to explain individual predictions.  
- **SHAP (SHapley Additive exPlanations)** was utilized for feature importance analysis.  

---

## 🎯 Deployment: Streamlit App  
Finally, I deployed our trained model using **Streamlit** to allow users to classify news articles interactively.  

🔗 **Live Demo:** [Streamlit Application](<your-streamlit-link-here>)  

---

### 🔎 Summary in One Line:  
💡 **NLP-powered multi-class classification achieved 95% accuracy with XGBoost, leveraging advanced text preprocessing, feature engineering, and model interpretability techniques.** 🚀🔥  

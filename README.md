# Amazon Product Recommendation using SVD
This project implements a collaborative filtering approach to build an Amazon product recommendation system using Singular Value Decomposition (SVD). The recommendation system is designed to provide personalized product suggestions to users based on their historical interactions with products. By leveraging advanced machine learning techniques, this system aims to enhance the user experience on Amazon by suggesting relevant products that users are likely to be interested in, thereby increasing user engagement and satisfaction. The project demonstrates the effectiveness of using SVD in a real-world e-commerce context, providing a robust framework that can be scaled and applied to other recommendation tasks.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Overview
Amazon, as one of the largest e-commerce platforms, hosts an extensive catalog of products and customer reviews. With millions of users interacting with the platform daily, providing personalized product recommendations becomes essential to enhance the shopping experience. This project focuses on analyzing customer interactions to recommend products using matrix factorization techniques like SVD. By leveraging user-item matrices, the system predicts user preferences for products they have not yet interacted with, thus enhancing the shopping experience and increasing customer satisfaction.

Key points:
- **Personalization:** The project aims to offer personalized product suggestions based on historical user-item interactions.
- **Enhanced User Experience:** By predicting user preferences, the system improves the shopping experience and increases customer satisfaction.
- **Matrix Factorization:** Utilizing techniques like Singular Value Decomposition (SVD) to identify latent factors influencing user preferences.
- **Scalability:** The framework can be scaled and applied to other recommendation tasks in e-commerce.

By implementing SVD, the project aims to demonstrate the power of collaborative filtering in making relevant product suggestions. This approach allows for the identification of patterns and trends in user behavior, providing a more accurate and personalized recommendation compared to traditional methods. Ultimately, the project seeks to contribute to a more efficient and enjoyable shopping experience on Amazon.

## Dataset
The dataset used in this analysis consists of Amazon product reviews dataset collected from Kaggle and avaliable to use in `data` folder. 

The data includes:
- User IDs
- Product IDs
- Ratings
- Timestamps

### Preprocessing Steps
- Removal of duplicate entries
- Filtering of sparse user-product interactions
- Normalization and preparation for SVD

## Approach
The project utilizes a collaborative filtering-based recommendation strategy:
1. **Data Preparation:** Creating a user-item interaction matrix.
2. **Matrix Factorization with SVD:** Decomposing the user-item matrix into lower-dimensional latent factors.
3. **Prediction:** Reconstructing the matrix to estimate missing values (unrated items).
4. **Evaluation:** Measuring performance using metrics such as RMSE.

### Popularity-Based vs. Collaborative Filtering
Popularity-based recommender systems rely on frequency counts and are not tailored to individual users, which may not provide the best recommendations. For example, the Popularity-based model might give the same set of five products to different users, whereas the Collaborative Filtering-based model provides personalized recommendations based on each user's previous purchase history.

Model-based Collaborative Filtering, such as the approach used in this project, is personalized and makes recommendations based on the user's historical behavior without needing any additional data.

## Technologies Used
The project is implemented using:
- **Programming Language:** Python
- **Libraries and Frameworks:**
   - scikit-learn
   - Pandas
   - Matplotlib
   - Seaborn
   - NumPy
   - surprise (for SVD implementation)
   - 
- **Jupyter Notebook:** For analysis and visualization

## Usage
To run this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshaEadara/Amazon-Product-Recomendation-using-SVD.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Amazon-Product-Recomendation-using-SVD
   ```
3. Install Dependencies:
Make sure you have Python installed. Then install the required libraries:
   ```bash
   pip install requirements.txt
   ```
4. Run the Notebook:
Open the Jupyter Notebook and execute the cells
   ```bash
   jupyter notebook Analysing_Flipkart_Reviews_using_Sentiment_Analysis_.ipynb
   ```
5. Run the cells sequentially to execute the analysis.

## Results
The recommendation model was evaluated using RMSE to measure prediction accuracy. The final results demonstrate the effectiveness of the SVD-based collaborative filtering approach in capturing user preferences.

**Key Observations:**
- SVD efficiently identifies latent factors representing user-product relationships.
- The model achieves competitive RMSE scores, demonstrating its reliability for recommendation tasks.
- Model-based Collaborative Filtering provides personalized recommendations by leveraging users' historical behavior without needing additional data.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork this repository, make changes, and submit a pull request. Please ensure your code adheres to the project structure and is well-documented.

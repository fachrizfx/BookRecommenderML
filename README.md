# Machine Learning Project Report
Project resources that have been used to help build this model are listed below in the reference area. Please cite this GitHub Repository page if you used our model as a guidance reference. Images that have been used in this markdown may not be rendered due to one or another reason, please try refreshing the page to see the image rendered.

## Project Overview

We certainly often hear the phrase "Books are windows to the world", maybe how many of us already know the meaning of that sentence. But for those who haven't, this sentence illustrates how important books are because books are an endless source of knowledge. According to [11] reading books has many benefits, one of which is that it can prevent cognitive decline caused by age. Therefore we must cultivate a culture of reading. We can read through physical books or digital books / E-Books. In the field of Machine Learning, we can contribute to cultivating a reading culture in many ways, one of which is by creating a recommendation system so that we can all read according to what we like.

## Business Understanding

From the explanation in the Project Overview section, we know that a recommendation system can help cultivate a culture of reading books. But for business people what are the benefits? The answers can vary, one of which is that if we have a bookstore business, we can develop a recommendation system that can recommend book users/customers based on what category of books they like, so that the bookstore business can also benefit from the sale of books. recommended by the system.

### Problem Statements

Of course, we must have a goal. To make this goal we must have a problem, or in this project, it is as follows:
- How to make a recommendation system with the **content base filtering** technique?
- How is the suitability of the book recommendations given to the suitability of the user?
- How can a recommendation system help in the business sector?

### Goals

We can make goals from the problems above. The goals for this project are as follows:
- Create a recommendation system based on user preferences
- Knowing the suitability of recommendations to users
- See if the model/system is suitable for use in the business field

### Solution statements

In order to be able to answer the problems described above, we can create a solution as follows:
- Implementing personalized recommendation system techniques
- See if the categories of books we've read are listed in the recommendations
- Evaluate the results of the second solution statement to determine whether it is suitable for use in the business field

## Data Understanding

![insightdata](https://drive.google.com/uc?export=view&id=1YYGyqpuzy-I7fKDSJnktaX3U6162Mi1y)

The dataset that I will use in this project has 2 folders in the directory, namely: 'Book reviews' and 'Books Data with Category Language and Summary', here I will use the second folder, namely 'Books Data with Category Language and Summary' '. Inside this folder we will find a file called 'Preprocessed_data.csv', this file contains a summary of the first folder. The amount of data contained in this file is also quite large, namely 1031175. To view or download the dataset, you can go through the following link: [Kaggle]

The variables in the Book-Crossing: User review rating dataset are as follows:
- user_id: user-id
- location: user location
- age: user age
- ISBN: the book's ISBN code
- rating: rating the book
- book_title: the title of the book
- book_author: the author/author of the book
- year_of_publication: year of publication
- publisher: book publisher
- Summary: book summary
- Language: the language of the book
- Category: book category's
- city: city from user location
- state: state of user's location
- country: country of user's location

Before we get to the Univariate Analysis stage, we will drop the rows 'Unnamed: 0', 'img_s', 'img_m', 'img_l', 'Summary', 'location' because those columns are not very useful. The reason the 'location' column is also dropped is that the column is already represented by the 'city', 'state', 'country' columns.

## Univariate Analysis
Before we enter the Data Preparation stage, we can enter the Data Analysis stage first to find out the data in this dataset. At this stage, I will use the Univariate Data Analysis technique. This technique is the most basic technique for analyzing data. In short, Uni means one, which means analyzing data separately (one by one). Its purpose is to view and provide insight into our data.

![analysis results](https://drive.google.com/uc?export=view&id=181cwBBJIxJ1KR0wxl5xlGAeTXNpNRj7g)

We can conclude that the 'city', 'state', and 'country' columns have values ​​that are NaN, this means that these columns have missing values. If we look again we can see that in the column 'Language', and 'Category' there is also a missing value which is represented by the value '9', this value can be said to be a missing value because it is impossible for the column 'Language', and 'Category' to have a value. which is '9'. The number of columns that have a value of '9' is also quite large, namely 176176. We will handle all these missing values ​​at the **Data Cleaning** stage.

## Data Preprocessing
At the Data Preprocessing stage here I will not do any techniques for Data Preprocessing because the dataset I use is already combined, or you could say it is ready to use and only needs to do Data Cleaning of missing values. Therefore, here I will not do anything other than performing the Sorting technique with the .sort_values [01] function from the Pandas library.

## Data Preparation
At this stage, we will prepare the data for training. Data preparation here includes Data Cleaning and Feature Selection. Let's start with the first step which is Data Cleaning.

### Data Size Reduction
I wrote this paragraph at the Modeling stage. So when I wanted to calculate the cosine similarity of this project I found a problem, namely my runtime keeps crashing because the RAM usage exceeds the maximum limit, if I activate the GPU hardware accelerator, an error will appear that the runtime cannot connect to the GPU on Back-end. This could be because I often use the GPU so according to [10] "As a result, users who use Colab for long-running computations, or users who have recently used more resources in Colab, are more likely to run into usage limits and have their access to GPUs and TPUs temporarily restricted" or it can be concluded that if we use more computing resources, we will tend to experience usage limits, or there are other factors that cause this. Therefore, here I decided to reduce the dataset size by dropping > 50%. I've tried reducing < 40% but the result is still the same i.e. runtime crash.

### Data Cleanup
As mentioned in the Univariate Analysis section, our data has quite a lot of missing values. Therefore, at this stage, we will deal with missing values.

#### Handling Missing Value
We already know that the missing values ​​are in the 'city', 'state', 'country', 'Language', and 'Category' columns. The two 'Language' and 'Category' columns have missing values ​​that have been filled/represented with the value '9'. This missing value is not detected by the .isnull() function of the Pandas library [02]. This is because the .isnull() function only detects null values. The null value is a value that has no value. The missing value in the second column is represented by '9' which is of type Object/String, therefore the missing value is already represented by any value in the string. The technique according to [03] is called the 'Imputation Method for categorical columns' or if translated the imputation method for categorical columns. According to [03] one of the advantages of this technique is that it prevents data and the disadvantage is that it can make performance decrease during the encoding process.

According to [03] also, one way to deal with this is to replace it with the category that occurs most frequently. But I don't think this is suitable because we recommend books based on their categories. For example, imagine that we like to read books in the Technology category, and in an E-book application/website, we must be talking about Action books. It is more suitable for input in the exploration section rather than the recommendations section for you, so the books recommended by the system are not a match for what we like. Therefore we will use the Delete Rows with Missing Value technique using the .dropna() [04] function from the Pandas library.

Before we delete rows with missing values ​​we need to remember that the missing values ​​in the column 'Language', and 'Category' are represented by the proper string '9' therefore we have to convert the value '9' to NaN so that it can be dropped by the .dropna function (). We can replace that value with the .replace() function from the Pandas library [05].

Output:
```
user_id                0
age                    0
isbn                   0
rating                 0
book_title             0
book_author            0
year_of_publication    0
publisher              0
Language               0
Category               0
city                   0
state                  0
country                0
dtype: int64
```

```
after data cleaning dataset size: 303428
```

After we clean we can see that our data is clean, but we experience a data loss of more than 40%, which is quite significant. Next, we will delete all duplicate ISBNs, this is done so that there are only unique books. We will drop it with the .drop_duplicates() function [06].

### Feature Selection
At this stage, we will select features that are relevant to our target variable.

#### Separating Features from Dataset
First of all, we have to convert the data into a list first. We can do this conversion with the .tolist() function [07] from the NumPy library. Before separating the features, we need to select the features that will be used first. In this project I will use the columns 'ISBN', 'book_title', 'book_author', 'publisher', 'Language', and 'Category'.

After transforming it into a list, we can create a data frame with Pandas. To create it here we will use a dictionary to pair it with key values. The results are as follows:

![featuredataset](https://drive.google.com/uc?export=view&id=1CDoN2wdVi3iXNHBFv6j_LE_UmuhxvXmb)

The next stage is the long-awaited stage, namely Modeling.

## Modeling
As we know the output of the recommendation system is *Top-N* which means it is not like classification, regression, etc. models which only have one output which is a prediction. It is different from the recommendation system which presents many outputs. An example of a recommendation system is Youtube, the videos recommended by Youtube are examples of Top-N output. Why does the recommendation system provide Top-N output? The answer is very simple, namely, it is impossible for the system to recommend only one thing, this is because if we don't like the recommendation, what is the next recommendation? So the answer is so that we can choose which one we like.

### Vectorizer
First of all, we have to start with Vectorizer, in this project, we will use CountVectorizer from [08] Scikit Learn library. Actually, we can use the TF-IDF Vectorizer but here I use the CountVectorizer because TF-IDF works by scoring a word, words that often appear and rarely appear will be given a different score, the score serves to determine the meaning or context of a sentence. In this case, we do not need to understand the meaning of the text, because here we only need to obtain as much information as possible to calculate the degree of similarity with cosine similarity. Therefore here we will use the CountVectorizer.

In this case, I will use Category as a reference to recommend books. The question is, what is the purpose of a vectorizer? The purpose of the vectorizer in the recommendation system is to find the right representation to represent the category.

Before we start, of course, we have to import the CountVectorizer library [08] first, but since I've imported all libraries at the beginning of the cell, we don't need to re-import.

First of all, we have to start with the Vectorizer, in this project, we will use the CountVectorizer [08] from the Scikit Learn library. In this case, I will use Category as a reference to recommend books. The question is, what is the purpose of a vectorizer? The purpose of the vectorizer in the recommendation system is to find the right representation to represent the category. Next, we can represent the CountVectorizer with a variable named Vec. Then the vectorizer will convert the text set into a matrix of the number of tokens in the category column so that the vectorizer converts text we must use .fit(). After everything is done we can transform it into a matrix form, and in order for the vector to become a matrix form, we have to transform again into a dense with the .todense() function.

### Cosine Similarity
Cosine similarity calculates the degree of similarity between each category. We can calculate cosine similarity with the function cosine_similarity() [09] from the Scikit Learn library. The output of this function is a matrix array so that we can present it in the form of a data frame.

![cosdataframe](https://drive.google.com/uc?export=view&id=1v9-0-H3T6c2CSqkvqN2ZOtlgQNItf1PB)

We can create a function to output Top-N. To use this function, we can use one of the titles from the book, and then we can choose the Top-N value which is represented by K. Here is an example of the Top-N output:

![recommendations](https://drive.google.com/uc?export=view&id=17QxLvxrfOYzM2yesO4OTbiCYNAEU1hMr)

## Evaluation
We have created a model to recommend books with Content-Based Filtering techniques. Now we are in the evaluation stage. Now we can answer all the problems described in the Problem Statement section:

- How to make a recommendation system with the **content-based filtering** technique?
- How is the suitability of the book recommendations given to the suitability of the user?
- How can a recommendation system help in the business sector?

We have created a recommendation system with the Content-Based Filtering technique. Now we can get recommendations with the get_recommendations() function. If we test the function with the book title 'Dragonshadow' we can see the output as follows:

![dragonshadowcategory](https://drive.google.com/uc?export=view&id=1MLLLVpqO-QwyWZds-O0SxTk38pIl2MZE)

We can see that the book belongs to the category 'Fiction' or fiction, then the results recommended by the model are also category fiction, this indicates a match between user preferences and recommended ones. So we can conclude that the recommended book is suitable for the user so that the user can buy the book so that our bookstore business can benefit from users who like the results of the system recommendation.

Next, we will measure precision with Precision Metric. We can obtain this metric by dividing the relevant recommendations by the number of recommendations. The question is how do we know which recommendations are relevant and which are not? To find out, we can see from the recommendation category whether it matches the category we wrote as Input. In this case, we can see in the image above that all of our recommendations have the same category, namely Fiction.

We will use Precision Metric to measure the precision of the recommendation system. This metric works by dividing the number of relevant recommendations by the number of recommendations and then we multiply by 100 to make it a percentage, or mathematically written as follows:

```
precision percentage = relevant / number of top-n * 100
```

Technically this metric works by comparing the number of relevant recommendations with the total number of recommendations so that we can get a comparison. For example imagine we have the number of relevant recommendations of 9 and our total number of recommendations is 13, so we get a 9/13 comparison. As we know to make it a percentage we need to multiply the ratio by 100 so we have the final form 9/13*100, then we can apply basic math arithmetics to get a precision output in the form of a percentage of 69.2%.

We can create a function to measure precision using Precision Metric. The function we created will require two inputs, including the Number of Relevant Recommendations and the Number of Top-N. The inputs are then divided (according to the formula above) and then multiplied by 100 to become a percent unit. Here is one of the output percentages of the system precision:

```
Recommendation System Precision Percentage: 100.0%
```

We can see that our system precision reaches 100%! So far, we have seen that our system can recommend books that are relevant to the preferences of the user/customer.

**All parts of the document are translated to English**

# Reference List

<br />[[Kaggle]] Bhatia, R. (2021, February 17). Book-Crossing: User review ratings (Version 3) [A collection of book ratings]. Kaggle. https://www.kaggle.com/ruchi798/bookcrossing-dataset
<br />[[01]] Pandas Pydata. (n.d.-c). pandas.DataFrame.sort_values. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
<br />[[02]] Pandas Pydata. (n.d.-c). pandas.isnull. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.isnull.html
<br />[[03]] Kumar, S. (2020, July 24). 7 Ways to Handle Missing Values in Machine Learning. Towards Data Science. Retrieved January 16, 2022, from https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e
<br />[[04]] Pandas Pydata. (n.d.-b). pandas.DataFrame.dropna. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
<br />[[05]] Panda Pydata. (n.d.). pandas.DataFrame.replace. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
<br />[[06]] Pandas Pydata. (n.d.). pandas.DataFrame.drop_duplicates. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html
<br />[[07]] NumPy. (n.d.). numpy.ndarray.tolist. Retrieved January 16, 2022, from https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
<br />[[08]] Scikit Learn. (n.d.-a). sklearn.feature_extraction.text.CountVectorizer. Retrieved January 16, 2022, from https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
<br />[[09]] Scikit Learn. (n.d.). sklearn.metrics.pairwise.cosine_similarity. Retrieved January 16, 2022, from http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
<br />[[10]] Google. (n.d.). Colaboratory FAQ. Google Research. Retrieved January 16, 2022, from https://research.google.com/colaboratory/faq.html
<br />[[11]] Bola. (2021, February 16). 7 Manfaat Membaca Buku yang Masih Belum Banyak Diketahui. Retrieved January 16, 2022, from https://www.bola.com/ragam/read/4484476/7-manfaat-membaca-buku-yang-masih-belum-banyak-diketahui

<br />
<br />

[Kaggle]: https://www.kaggle.com/ruchi798/bookcrossing-dataset
[01]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
[02]: https://pandas.pydata.org/docs/reference/api/pandas.isnull.html
[03]: https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e
[04]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
[05]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
[06]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html
[07]: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
[08]: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
[09]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
[10]: https://research.google.com/colaboratory/faq.html
[11]: https://www.bola.com/ragam/read/4484476/7-manfaat-membaca-buku-yang-masih-belum-banyak-diketahui

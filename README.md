# gdsc_ml_grade_predict
This is done for this particular task.
Task 1 (Beginner - 3 Points)
Title: AI-Based Grade Prediction
Description:
Build a model to predict a student’s final grade based on features such as
attendance, participation, assignment scores, and exam marks.
Dataset:
Use the Student Performance Dataset or synthesize a similar dataset. You are
free to search for and use better datasets.
Deliverables:
1. Train an ANN or Decision Tree model for regression or classification.
2. Evaluate using RMSE or classification metrics (e.g., accuracy, F1-score).
3. Analyze feature importance to identify key contributors to performance.
Bonus (Productization):
Build a web app ( Streamlit is fine ) where users can input student details to
predict grades.

I have used kaggle.I have to use by the kaggle input directory to upload the input files, which I have also uploaded to the GitHub Repo you can view the csv file in the repo that I Used which was provided in the link for the induction page.
This link will direct you to the dataset.
https://archive.ics.uci.edu/dataset/320/student+performance

I use decision tree regressor over here and also to get how much error is there that is also done. I collected the types and everything and then I am setting the train test into 0.2 size the test size is 0.2 and train is 0.8 so it is pretty standard exercise 80:20 ratio for training the model and then the regular modify the model. To train using the decision tree regressor which was suggested in the induction pdf.We are predicting from the test at whatever like the rest testing we had separated.

Then we are checking the error score(root mean square) and how much the differences and then we are printing it. You can see in the screenshots that I have uploaded.You can see how much error is there at the end.

Before that yes we need to optimise the input data and optimising would mean removing some features like we can actually optimise it better creating training set separately for urban and rural would be one important thing because they are pretty different and then internet access at home is a important thing now and yeah that will be play with the social aspect.
We can find some aspects and other features of the data to get better results. Also training the data we use separate training methods for different data sample set.The usual thing would be to separate the male and female.

As an example for different School I think it would be better to train different model to create two models and in one model based on the one school which in my taken data set as given as GD and then another school is it so we will train the data which comes under one school into one training model and then the other into another training model and when we are taking the input we check the input based on which school that particular student went to. This helps because these are two separate things and as I am using a decision tree regressor mixing it up ie; if I am using all the all the different parameters into one model it's better to split the model into two and then enter it. 

There are many other factors also which we can use first, we would be first checking the statistics of the input. How much the input varies with. When we have system like 1 to 5 it's pretty easy because we can just scale it but when we have some drastic unrelated things like when we are using Urban and rural and when we are taking mother's job and all it would be better to create separate totally separate instead of a gradient. It would be better to make a different model type of thing.

After that I have taken the user input I will be uploading screenshot. I have made a separate example and it's predicted final grade.

All the screenshots are uploaded in the repo.

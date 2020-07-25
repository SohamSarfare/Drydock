# DryDock Questions
This is a trial python project which tries to help out my favourite naval historian Drachinifel in sorting out questions for his Drydock series.  

Below each video of his, there is one pinned comment to which anyone can reply and ask any naval question that they have in mind. The idea of this project is to scrap the replies made to this pinned comment on every video and then classify whether the reply made is a question or not. Also, the project aims to ompile all the questions already answered in previous drydock questions by scrapping the descriptions of videos. Finally, the entire project is to be made user friendly by operating as a discord bot with easy to use commands

## Creating Archive
Instead of going through each and every Drydock video to check if your question has been asnwered, one can simply check this archive (which is in .xlsx format) and save a lot of time. Drachinifel makes this easier by noting down the questions answered in the description text of the particular video. The idea hence, is to get the description text of the videos and note down the timestamp and question from it. 
1. Get the description text using youtube api.
2. Split the text into seperate lines using `.split('\n')`. These new lines are stored in a pandas dataframe. 
3. Empty lines are removed by using the `.dropna()` method of pandas.
4. Now, after a certain number of questions, we have other channel realted stuff like social links. To get rif od this, we use a simple regex flag. Once the flag is reached, the rows after the flag are discarded. 

## Scrapping Comments
Scrapping the comments is being done using Youtube API. The logic being:
1. Search for the Youtube channel 'Drachinifiel'.
2. Look for the playlist 'Uploads' on finiding the channel.
3. When in uploads, load the first video and look for a comment begining with 'Pinned post for'.
4. Extract the comment ID of this commentsing this as the parent ID, extract all the child comments. 
5. Save the child comments along with the username of the commentor and the video that the comment was made in a pandas dataframe.
6. Export this dataframe as a .xlsx file to use in further analysis.

## Classification
Since this is a binary choice i.e. questions or not, Support Vector Machines prove to be the best classifier. Certain unique cleaning steps taking to make classification easier are:  
1. Subtituting hyperlinks made in replies by the word 'hyperlink'.  
2. Whenever a user replies to a child comment, the '@' prefix is used. These replies are generally answers to the child comment. Hence '@' is substituted by 'answer' in these replies.  
3. The '?' sign is replaced by the word 'question'.  
After this, stopwords are removed and a vocabulary is built of the remaining words in the dataset. This vocabulary is used to convert the comments into TF-IDF transformed vectors. To train the support vector classifier, a subset of all the comments (about 1000) are manually tagged in the file 'questions_trainer.xlsx'. After training on this file, the SVC can be used to classify the entire dataset. SVC obtained a training accuracy of 0.89.

## Discord Bot
To make the program easier to use, I considered turning it into a discord bot using `discord.py`. Using a single command the bot
### Usage: 
1. To get archive of questions a single command has been assigned `!archive` with certain aliases. 
2. To get comments classified as questions or not, a command `!getquestions *` has been assigned. Here, `*` implies that an argument is needed. 
3. To get classified comments from all videos, the aargument `all` should be used. To get classified comments from a particular video, the url of the video should be passed.

I have implemented a check system which checks whether the video is made by Drachinifel or not. If it isnt, then the bot returns an appropriate error. 
## Working
When the `getquestions` command is invoked, the scrapping script is executed. The output from this is passed onto the classification script. This script cleans and classifies the data and returns the .xlsx file to the bot which uploads it to the discord server.  

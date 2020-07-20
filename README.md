# DryDock-Questions

This is a trial python project which tries to help out my favourite naval historian Drachinifel in sorting out questions for his Drydock series.  

Below every video of his, there is one pinned comment to which anyone can reply and ask any naval question that they have in mind. The idea of this project is to scrap the replies made to this pinned comment on every video and then:  
1. Classify whether the reply is a question or not.  
2. If the reply is a question, check if the question has been asked in previous videos or not.
3. If it has been asked before, has the question been answered by Drachinifiel in one of his videos or not.

# Scrapping Comments

Scrapping the comments is being done using Youtube API. The logic being:
1. Search for the Youtube channel 'Drachinifiel'.
2. Look for the playlist 'Uploads' on finiding the channel.
3. When in uploads, load the first video and look for a comment begining with 'Pinned post for'.
4. Extract the comment ID of this commentsing this as the parent ID, extract all the child comments. 
5. Save the child comments along with the username of the commentor and the video that the comment was made in a pandas dataframe.
6. Export this dataframe as a .csv file to use in further analysis.

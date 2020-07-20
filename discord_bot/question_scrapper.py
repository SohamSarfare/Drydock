# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:42:51 2019

@author: Soham
"""
import config
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import re
import os.path

DEVELOPER_KEY = config.credentials['dev_key']
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def getQues(youtube, video_id, questions, exists):
    try:
        # Getting video details
        video_request = youtube.videos().list(part="snippet", id=video_id)
        video_res = video_request.execute()
        videoTitle = video_res.get("items")[0].get("snippet").get("title")
        print("Scrapping Video: {}".format(videoTitle))
        # Getting top comment ID
        top_request = youtube.commentThreads().list(part="snippet", videoId=video_id,
                                searchTerms="Pinned post for", textFormat="plainText")
        top_res = top_request.execute()
        top_comment = top_res.get("items")[0].get(
                "snippet").get("topLevelComment").get("id")

            # Getting replies to top comment
        comments_request = youtube.comments().list(part="snippet", parentId=top_comment,
                                                       textFormat="plainText", maxResults=100)
        comments_res = comments_request.execute()
            
        for comment in comments_res.get("items"):
            temp = comment.get("snippet").get("textDisplay")
            ques = re.sub('[,]', ' ', temp)
            ques = re.sub('[=]',' ',ques)
            user = comment.get("snippet").get("authorDisplayName")
            if exists:
                questions = pd.DataFrame(data={"User": user, 
                                                "Question": ques.lower(), 
                                                "Video": videoTitle, 
                                                "VideoID": video_id}, index=[0]).append(questions)
            else:
                questions = questions.append(pd.DataFrame(
                    data={"User": user, "Question": ques.lower(), "Video": videoTitle, "VideoID": video_id}, index=[0]))
        questions = questions.reset_index(drop=True)
        return questions
    except IndexError:
        return questions 

def lastVideo():
    questions = pd.DataFrame(columns=["User", "Question", "Video", "VideoID"])
    exists = False
    """ if os.path.isfile(r'files\questions.xlsx'):
        questions = pd.read_excel(r'files\questions.xlsx')
        exists = True """
    youtube = googleapiclient.discovery.build(
    YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    
    channel_request = youtube.channels().list(
        part="contentDetails", forUsername="Drachinifel")
    channel_res = channel_request.execute()
    uploads = channel_res.get("items")[0].get(
        "contentDetails").get("relatedPlaylists").get("uploads")
    #last_video_id = 0
    """ try:
        last_video_df = pd.read_csv(r'files\last_video.csv')
        last_video_id = last_video_df['last_video'][0]
        print('Last video scrapped: {}'.format(last_video_id))
    except:
        last_video_df = pd.DataFrame(data={'last_video': 0}, index=[0])
        last_video_df.to_csv(r'files\last_video.csv')
        last_video_id = 0 """
    
    playlist_request = youtube.playlistItems().list(
        part="contentDetails", playlistId=uploads, maxResults=50)
    playlist_res = playlist_request.execute()

    #start_scrapping = True
    while("nextPageToken" in playlist_res):
        for video in playlist_res.get("items"):
            video_id = video.get("contentDetails").get("videoId")
            """ if video_id == last_video_id:
                start_scrapping = False
                break
            elif start_scrapping: """
            """ print(start_scrapping)
                print(type(last_video_id))
                print(type(video_id))
                print(video_id)
                print(last_video_id) """
            questions = getQues(youtube, video_id, questions, exists)           
    
        nextToken = playlist_res.get("nextPageToken")
        playlist_res = youtube.playlistItems().list(part="contentDetails",
                                                    pageToken=nextToken, playlistId=uploads, maxResults=50).execute()
    last_video_df = pd.DataFrame(data= {'last_video': questions['VideoID'][0]}, index=[0])
    last_video_df.to_csv(r'files\last_video.csv')
    #questions.to_csv(r"files\questions.csv", index= False)
    questions.to_excel(r"files\questions.xlsx", index=False)
    print(r"Scrapped comments saved to files\questions.csv")
    #return questions
    
def scrapVideo(video_id):
    questions = pd.DataFrame(columns=["User", "Question", "Video"])
    youtube = googleapiclient.discovery.build(
    YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    questions = getQues(youtube, video_id, questions, False)
    #questions.to_csv(r"files\questions_{}.csv".format(video_id), index=False)
    questions.to_excel(r"files\questions_{}.xlsx".format(video_id), index=False)
    print(r"Scrapped comments saved to files\questions_{}.xlsx".format(video_id))
    #return questions
#lastVideo()

def check(video_id):
    youtube = googleapiclient.discovery.build(
    YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    video_request = youtube.videos().list(part="snippet", id=video_id)
    video_res = video_request.execute()
    author = video_res.get("items")[0].get("snippet").get("channelId")
    if author != 'UC4mftUX7apmV1vsVXZh7RTw':
        return False
    elif author == 'all':
        return True
    else:
        return True
    

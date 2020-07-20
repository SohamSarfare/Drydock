import config
try:
    import googleapiclient.discovery
    import googleapiclient.errors
    import pandas as pd
    import numpy as np
    import re 
except:
    import os
    os.system('cmd /c pip install pandas')
    os.system('cmd /c pip install numpy')
    os.system('cmd /c pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib')
    import googleapiclient.discovery
    import googleapiclient.errors
    import pandas as pd
    import numpy as np
    import re
    
DEVELOPER_KEY = config.credentials['dev_key']
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def get_description(youtube, playlist_res):
    try:
        descriptions = []
        titles = []
        for item in playlist_res.get("items"):
            # Getting video details
            videoID = item.get("contentDetails").get("videoId")
            video_request = youtube.videos().list(part="snippet", id=videoID)
            video_res = video_request.execute()
            videoTitle = video_res.get("items")[0].get("snippet").get("title")
            print("Scrapping Video: {}".format(videoTitle))
            video_description = video_res.get("items")[0].get("snippet").get("description")
            descriptions.append(video_description)
            titles.append(videoTitle)
        data = pd.DataFrame(data= {
            'title': titles,
            'description': descriptions
        })
        return data
    except IndexError:
        data = pd.DataFrame(data= {
            'title': titles,
            'description': descriptions
        })
        return data

youtube = googleapiclient.discovery.build(
    YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

drydock = 'PLMK9a-vDE5zHFYcpLUheMBT744l3ThhPb'

playlist_request = youtube.playlistItems().list(
    part="contentDetails", playlistId=drydock, maxResults=50)
playlist_res = playlist_request.execute()

data = get_description(youtube, playlist_res)

while("nextPageToken" in playlist_res):
    nextToken = playlist_res.get("nextPageToken")
    playlist_res = youtube.playlistItems().list(part="contentDetails",
                                                pageToken=nextToken, 
                                                playlistId=drydock, 
                                                maxResults=50).execute()
    data = data.append(get_description(youtube, playlist_res))

def store_question(epno, description):
    epno = re.findall(r'\d+', epno)
    questions = pd.DataFrame(data={
        'Episode': 'Ep {}'.format(epno[0]),
        'Question': description.split('\n')
    })
    questions['Question'].replace('', np.nan, inplace=True)
    questions.dropna(subset = ['Question'],inplace= True)
    count = 0
    for entry in questions['Question']:
        entry = entry.strip()
        if entry[0].isdigit():
            count+=1
        else:
            break
    questions = questions.iloc[:count,:]
    return questions

df = pd.DataFrame(columns= ['Episode', 'Question'])
for i in range(0, data.shape[0]):
    df = df.append(store_question(data.iloc[i,:]['title'],data.iloc[i,:]['description']))

df.reset_index(drop=True, inplace=True)
df.to_excel(r'files\Drydock Questions.xlsx', index= False)
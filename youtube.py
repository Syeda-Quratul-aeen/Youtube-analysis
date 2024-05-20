from googleapiclient.discovery import build
import pandas as pd
from IPython.display import JSON
from dateutil import parser
import isodate

#Data viz packages
import seaborn as sns
import matplotlib.pyplot as plt

#NLP
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

channels_id=['
             #add channels here
             ]

api_service_name = "youtube"
api_version = "v3"
def get_channel_stats(youtube, channels_id):

    """
    Get Channel stats

    Params:
    -----
    youtube: build object of youtube API
    channel_ids: Get channel IDs

    Returns:
    -----
    dataframe with all the channel stats

    """

    all_data=[]
    youtube = build(
        api_service_name, api_version, developerKey=api_key)

    request = youtube.channels().list(
            part="snippet,contentDetails,statistics",
            id=','.join(channels_id)
        )
    response = request.execute()

    #loop through the items
    for item in response['items']:
        data= {'channelName': item['snippet']['title'],
            'subscribers':item['statistics']['subscriberCount'],
            'views':item['statistics']['viewCount'],
            'totalViews':item['statistics']['videoCount'],
            'playlistId':item['contentDetails']['relatedPlaylists']['uploads']
            }
        all_data.append(data)

    return(pd.DataFrame(all_data))
channel_stats = get_channel_stats(youtube, channels_id)
print(channel_stats)

def get_video_ids(youtube, playlist_id):
    video_ids = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet, contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break  # Exit the loop if there are no more pages

    return video_ids

JSON(response)

video_ids=get_video_ids(youtube, playlist_id)

len(video_ids)

def get_video_details(youtube, video_ids): 
    all_video_info = []
    request = youtube.videos().list(
            part="snippet, contentDetails, statistics",
            id=video_ids[0:5]
        )
    response = request.execute()

    for video in response["items"]:
        stats_to_keep = {'snippet': ["channelTitle", 'title', 'description','tags', 'publishedAt'],
                        'statistics': ['viewCount', 'likeCount','favoriteCount' ,'commentCount'],
                        'contentDetails':['duration', 'definition', 'caption']}
        video_info = {}
        video_info["video_id"]= video['id']

        for k in stats_to_keep.keys():
            for v in stats_to_keep[k]:
                try:
                    video_info[v] = video[k][v]
                except:
                    video_info[v] = None
        all_video_info.append(video_info)
    return pd.DataFrame(all_video_info)

video_df = get_video_details(youtube, video_ids)

print(video_df)

def get_comments(youtube, video_ids):
    all_comments = []
    for video_id in video_ids:
        request = youtube.commentThreads().list(
            part="snippet, replies",
            videoId=video_id
        )
    response = request.execute()

    for item in response.get('items', []):
        comment = item['snippet']['topLevelComment']['snippet']
        if comment['likeCount'] > 10 and comment['canRate']:
            comments.append({
                'text': comment['textDisplay'],
                'author': comment['authorDisplayName'],
                'likes': comment['likeCount'],
                'publishedAt': comment['publishedAt']
            })
        all_comments.append(comment)
    return pd.DataFrame(all_comments)

comments= get_comments(youtube, video_ids)
print(comments)

# Data Preprocessing

video_df.isnull().any()
video_df.dtypes

#converting to numeric data type
numeric_cols = ['viewCount','likeCount', 'favoriteCount', 'commentCount']
video_df[numeric_cols]= video_df[numeric_cols].apply(pd.to_numeric, errors = 'coerce', axis=1)

video_df[numeric_cols].dtypes

#publish day in the week
video_df['publishedAt'] = video_df['publishedAt'].apply(lambda x: parser.parse(x))
video_df['publishDayName']= video_df['publishedAt'].apply(lambda x: x.strftime("%A"))

#Get video duration in seconds
video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x))
video_df['durationSecs'] = video_df['durationSecs'].astype('timedelta64[s]')

video_df[['durationSecs','duration']]

#add tag count
video_df['tagCount'] = video_df['tags'].apply(lambda x:0 if x is None else len(x))
print(video_df)

#EXPLORATORY DATA ANALYSIS

#Best performing videos
import matplotlib.ticker as ticker
ax = sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=False)[0:9])
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000) + 'K'))

#worst performing videos
import matplotlib.ticker as ticker
ax = sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=True)[0:9])
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000) + 'K'))

#views distribution per video
sns.violinplot(x='channelTitle', y='viewCount', data=video_df)
#plt.xticks(rotation=90)  
plt.show()

#Views vs. likes and comments
fig, ax = plt.subplots(1, 2, figsize=(12, 6))  

# First scatter plot for comment count vs. view count
sns.scatterplot(data=video_df, x='commentCount', y='viewCount', ax=ax[0], color='blue', edgecolor='none')
ax[0].set_title('Comment Count vs. View Count')  # Title for the first subplot
ax[0].set_xlabel('Comment Count')  # X-axis label
ax[0].set_ylabel('View Count')  # Y-axis label
ax[0].set_xscale('log')  # Using log scale if data is skewed
ax[0].set_yscale('log')

# Second scatter plot for like count vs. view count
sns.scatterplot(data=video_df, x='likeCount', y='viewCount', ax=ax[1], color='green', edgecolor='none')
ax[1].set_title('Like Count vs. View Count')  # Title for the second subplot
ax[1].set_xlabel('Like Count')  # X-axis label
ax[1].set_ylabel('View Count')  # Y-axis label
ax[1].set_xscale('log')  # Using log scale if data is skewed
ax[1].set_yscale('log')

# Automatically adjust subplot params so that the subplot(s) fits in to the figure area
plt.tight_layout()

# Show the plot
plt.show()

#Video duration

# Creating a histogram of the 'durationSecs' column
plt.figure(figsize=(10, 6))  
hist_plot = sns.histplot(data=video_df, x='durationSecs', bins=10, color='skyblue', edgecolor='black')

# Adding title and labels
plt.title('Distribution of Video Duration')  # Add a title to the histogram
plt.xlabel('Duration in Seconds')  # Label for the x-axis
plt.ylabel('Number of Videos')  # Label for the y-axis

# Display the plot
plt.show()

#Wordcloud for video titles
stop_words = set(stopwords.words('english'))
video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])

all_words = list([a for b in video_df['title_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words)

def plot_cloud(word_cloud):
    plt.figure(figsize=(30,20))
    plt.imshow(word_cloud)
    plt.axis("off");

wordcloud = WordCloud(width= 2000, height=1000, random_state=1, background_color='black',
                      colormap='viridis', collocations = False).generate(all_words_str)
plot_cloud(wordcloud)

#Upload Schedule
day_df = pd.DataFrame(video_df['publishDayName'].value_counts())

# Making sure all days of the week are included even if some days have zero videos
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_df = day_df.reindex(weekdays, fill_value=0)  # fill_value ensures that missing days are set to zero

# Resetting the index to turn it into a column
day_df = day_df.reset_index()
day_df.columns = ['Day of the Week', 'Number of Videos']  # Renaming columns for clarity

# Plotting the bar graph
fig, ax = plt.subplots(figsize=(10, 6))  # Setting the size of the figure
bar_plot = ax.bar(day_df['Day of the Week'], day_df['Number of Videos'], color='skyblue')

# Adding title and labels
plt.title('Number of Videos Published Per Day')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Videos')

# Rotating x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.show()

import streamlit as st
import twint
#import nest_asyncio
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from typing import List, Any, Iterable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from urllib.request import urlopen


#with st.spinner('Loading tweets in app...'):
    
@st.experimental_memo
def tweets():

    #nest_asyncio.apply()

    c = twint.Config()

    c.Search = 'דיפלומט'       # topic
    #c.Limit = 500      # number of Tweets to scrape
    c.Store_csv = True       # store tweets in a csv file
    c.Output = "Data\\diplomat.csv"     # path to csv file

    c.Hide_output = True
    twint.run.Search(c)
    df = pd.read_csv('Data\\diplomat.csv', encoding= 'unicode_escape')
    return df


df = tweets()
    

def flatten(lst: List[Any]) -> Iterable[Any]:
    """Flatten a list using generators comprehensions.
        Returns a flattened version of list lst.
    """

    for sublist in lst:
        if isinstance(sublist, list):
            for item in sublist:
                yield item
        else:
            yield sublist

            

date = datetime.strftime(datetime.now() - timedelta(0), '%Y-%m-%d')

#title
st.write(f"""# Diplomat Twetter Analytics {date}""")
st.write('---')

date_select = st.date_input('Choose Date:')


    

def dip_imgs_daily(days_ago = 0):
    
    import warnings
    warnings.filterwarnings("ignore")
    date = datetime.strftime(datetime.now() - timedelta(days_ago), '%Y-%m-%d')
    cond = df['date']==date
    today_dip_cols = ["time","username","tweet","likes_count","photos","retweets_count","replies_count","hashtags"]
    photo_df = df[cond].drop_duplicates(subset=["tweet","photos"])[df["photos"]!=""][today_dip_cols]

    img_lst = photo_df["photos"].apply(lambda x:  x[2:-2]).values.tolist() 
    lengths = [len(img.split(', ')) for img in img_lst]
    img_lst = [i if len(i.split(", "))==1 else i.split(", ") for i in img_lst]


    user_lst = photo_df.username.tolist()
    user_lst = [[i]*j for i,j in zip(user_lst,lengths)]

    time_lst = photo_df.time.tolist()
    time_lst = [[i]*j for i,j in zip(time_lst,lengths)]

    likes_count_lst = photo_df.likes_count.tolist()
    likes_count_lst = [[f"likes: {i}"]*j for i,j in zip(likes_count_lst,lengths)]

    img_lst = [i.replace('\'','') for i in list(flatten(img_lst))]
    likes_count_lst = list(flatten(likes_count_lst))
    time_lst = list(flatten(time_lst))
    user_lst = list(flatten(user_lst))
    
    img_lst = [i for i in img_lst if len(i)>0]
    
    
    
    all_imgs_numpy = [np.asarray(Image.open(urlopen(img)).resize(size=(1200, 1200))) for img in img_lst]
    
    devider = 8
    n1,n2 = devider,max(int(round(len(all_imgs_numpy)/devider)),1)
    while n1*n2<len(all_imgs_numpy):
        devider-=1
        n1,n2 = devider,max(int(round(len(all_imgs_numpy)/devider)),1)
    
    
    if len(img_lst)>0:
        
        fig = plt.figure(figsize = (25,15))
        
        for i in range(len(all_imgs_numpy)):
            ax = fig.add_subplot(n1 , n2 , i + 1 , xticks = [] , yticks = [])
            ax.imshow(all_imgs_numpy[i] , cmap = 'gray')
            ax.set_title(f"{user_lst[i]} {time_lst[i]} {likes_count_lst[i]}", size = 15)
        title= "תמונות של ציוצים על דיפלומט"    
        plt.suptitle(f'{date} {title[::-1]}',size = 40)
        plt.savefig(f'Data\\{date}_{title}', facecolor='white', transparent=False)
        #plt.show()
        my_img = np.asarray(Image.open(f'Data\\{date}_{title}.png'))
        return my_img
    
    
    else:
        warnings.filterwarnings("default")
        return f"no images in {date}"





if st.button('Results:'):
    td = datetime.now().date() - date_select
    td = td.days
    #st.write(f'{td}')
    st.write(f"""## Stats for {date}""")
        
    try: 
        st.image(dip_imgs_daily(days_ago = td))
        
    except:
        st.write(f"{dip_imgs_daily(days_ago = td)}")
    
    st.write(f"""### most liked tweet: """)
    date = datetime.strftime(datetime.now() - timedelta(td), '%Y-%m-%d')
    cond1 = df['date']==date
    day_df = df[cond1]
    cond2 = day_df['likes_count']==day_df['likes_count'].max()
    most_liked_tweet = day_df[cond2]["tweet"].values[0]
    most_liked_tweet = most_liked_tweet.encode('UTF-8').decode('UTF_16')
    st.text(f"{most_liked_tweet} Likes: {day_df['likes_count'].max()}")
    
    
    
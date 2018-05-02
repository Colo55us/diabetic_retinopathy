import os
import pandas as pd
df = pd.read_csv('trainLabels.csv')
path = 'data/train'

li_images = os.listdir(path)
'''
li_images_sort = sorted(li_images)
print(len(li_images_sort))
print(df.count())

df_final = pd.DataFrame(columns=['image','level'])
images = list(df['image'])
print(images[:5])
print(li_images[0].split('.')[0])
for img in li_images:

    if img.split('.')[0] in images:
        level = df.loc[df['image']==img.split('.')[0],'level'].iloc[0]
        if level > 1:
            level = 1
        df_final = df_final.append({'image':img.split('.')[0],'level':level},ignore_index=True)
        print('image->{0}    level->{1}'.format(img,level))

df_final.to_csv('data/final_labels.csv',sep=',')
print('--completed--')

li_final = []
for img in li_images:
    img_temp = img.split('.')[0]
    li_final.append(img_temp)


df = pd.DataFrame(li_final)
df.to_csv('li_images.csv')
'''


for i,row in df.iterrows():
    if row['image']+'.jpeg' not in li_images:
        df = df.drop(i)
        print(row['image']+'.jpeg',' Dropped')
    
for i,row in df.iterrows():
    if row['level'] > 1:
        df.iloc[i].level=1
df.to_csv('check.csv',sep = ',')
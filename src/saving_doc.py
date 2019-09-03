file=[]

for input in news_list_initial.text:
    file.append(input)

new_df=pd.DataFrame()
new_df['document_id']=new_df.index
new_df['document_text']=file

new_df.to_csv('/home/anujc/Documents/data.csv',index=False)
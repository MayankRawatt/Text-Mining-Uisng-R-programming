//Using R
//For single text document
//document must be in text format .txt and save to working directory,check directry by command getwd()

getwd()
readLines("yourfilename.txt")
str(readLines("yourfilename.txt"))
paste(readLines("yourfilename.txt"),collapse=" " )

text <- paste(readLines("yourfilename.txt"),collpase = " ")
gsub(pattern = "\\W",replace = " ",text)
text2 <- gsub(pattern = "\\W",replace = " ",text)
gsub(pattern = "\\d",replace = " ",text2)
text2 <- gsub(pattern = "\\d",replace = " ",text2)
text2
tolower(text2)
text2 <- tolower(text2)
install.packages(tm)
library(tm)
text2
removeWords(text2,stopwords())
text2 <- removeWords(text2,stopwords())
gsub(pattern = "\\b[A-Z]\\b{1}",replace = " ",text2)
text2 <- gsub(pattern = "\\b[A-Z]\\b{1}",replace = " ",text2)
stripWhitespace(text2)
text2 <- stripWhitespace(text2)
install.packages("stringr")
library(stringr)
install.packages(wordcloud)
library(wordcloud)
str_split(text2,pattern = "\\s+")
textbag <- str_split(text2,pattern ="\\s+")
class(textbag)
textbag <- unlist(textbag)
class(textbag)
textbag
str(textbag)
getwd()
match(textbag,poswords)
!is.na(match(textbag,poswords))
sum(!is.na(match(textbag,poswords)))
!is.na(match(textbag,negwords))
sum(!is.na(match(textbag,negwords)))
score <- sum(!is.na(match(textbag,poswords)))-sum(!is.na(match(textbag,negwords)))
score
sd(score)
mean(score)
wordcloud(textbag)
wordcloud(textbag,min.freq=4)
wordcloud(textbag,min.freq=4,random.order=FALSE,scale=((3,0.5),color=rainbow(3))

\\ For multiple documents 

getwd()
\\your file path will be displayed
file.choose()
folder <- copy and paste the file path here YOURFILEPATH
folder
list.files(path=folder)
list.files(path=folder,pattern = "*.txt")
filelist <- list.files(path=folder,pattern = "*.txt")
filelist
paste(folder,"\\",filelist.sep = " " )
filelist <-paste(folder,"\\",filelist.sep = " " )
readLines()
typeof(filelist)
a <- lapply(filelist,FUN=readLines)
lapply(a,FUN=paste,collapse = " ")
mycorpus <- lapply(a,FUN=paste,collapse = " ")
mycorpus

gsub(pattern = "\\W",replace = " ",mycorpus)
mycorpus2 <- gsub(pattern = "\\W",replace = " ",mycorpus)
gsub(pattern = "\\d",replace = " ",mycorpus2)
mycorpus2 <- gsub(pattern = "\\d",replace = " ",mycorpus2)
mycorpus2
tolower(mycorpus2)
mycorpus2 <- tolower(mycorpus2)
rem_words <= c("define the words you want to remove that you don't want seperate by comma for ex "dog",cat","elephant",stopwords("english"))
removeWords(mycorpus2,rem_words,stopwords("english"))
mycorpus2 <- removeWords(mycorpus2,rem_words,stopwords("english"))
gsub(pattern = "\\b[A-Z]\\b{1}",replace = " ",mycorpus2)
stripWhitespace(mycorpus2)
mycorpus2 <- stripWhitespace(mycorpus2)
mycorpus2
wordcloud(mycorpus2)
wordcloud(mycorpus2,random.order=FALSE,color = rainbow(3))
corpus3 <- Corpus(VectorSource(mycorpus2))
corpus3
tdm <- TermDocumentMatrix(corpus3)
tdm
as.matrix(tdm)
tdm
data <- as.matrix(tdm)
\\Now export this data to your PC for visualization through power BI or any other platform,It will include the number of occurances of words
data
colnames(data)
colnames(data) <- c("give names","give name2" etc)
comparison.cloud(data)
install.packages("corrplot")
//import the data you saved and remove textual value in excel convert to excel 
install.packages("readxl")
library(readxl)
corrdata <- read_excel("your document address")
corrdata
rquery.cormat(corrdata)
install.packages("ggplot")
install.packages("ggplot2")
install.packages("ggraph")
library(ggplot)
library(ggplot2)
install.packages(heatmap)
library(heatmap)
cormat<-rquery.cormat(corrdata, graphType="heatmap")

\\Python Notebook
use that imported file named 'data' from the R and import here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
data = pd.read_csv('C:\\Users\\win 10\\Desktop\\data.csv')
data.head()
pearsoncorr = data.corr(method='pearson')
pearsoncorr
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.)
def heatmap(x, y, size):
    fig, ax = plt.subplots()
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
data1 = pd.read_csv('C:\\Users\\win 10\\Desktop\\data.csv')
columns = ['incident','nearmiss','unsafe','unsafecondition'] 
corr = data[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(x=corr['x'],y=corr['y'],size=corr['value'].abs())
import matplotlib.pyplot as plt
plt.matshow(data1.corr())
plt.show()
f = plt.figure(figsize=(19, 15))
plt.matshow(data.corr(), fignum=f.number)
plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)
plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
rs = np.random.RandomState(0)
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')
plt.style.use('ggplot')
plt.imshow(data.corr(), cmap=plt.cm.Reds, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(data.columns))]
plt.xticks(tick_marks, data.columns, rotation='vertical')
plt.yticks(tick_marks, data.columns)
plt.show()
import statsmodels.api as sm
import matplotlib.pyplot as plt
corr = data.corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()

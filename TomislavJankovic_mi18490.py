#!/usr/bin/env python
# coding: utf-8

# # Klasterovanje teksta u skupu podataka zaposlenih u programerskim firmama
# 

# ## Uvod <a class="anchor" id="intro"></a>
# <hr />
# 
# Moj zadatak se sastojao iz toga da primenim algoritme klasterovanja koji su se radili na 
# časovima kursa Istraživanje Podataka, i da na osnovu njih zaključim neke zaključke iz skupa podataka *employee_reviews*. U daljem tekstu ćemo proći kroz faze preprocesiranja, ukidanja *"nebitnih"* kolona, i primene algoritama za klasterovanje, kao što su *K-Sredina*, *DBSCAN* i *Hierarhijsko klasterovanje*. Dotaći ćemo se i ocena klasterovanja i zaključivanja optimalnog broja klastera za zadate algoritme(*koeficijent senke* i metoda *lakta*) Nakon toga, ćemo prikazati podatke i izneti bitne zaključke za zadati skup. Kao zadatu alatku za seminarski rad sam odabrao *Jupyter* svesku i jezik *Python*. Odabrao sam ovo okruženje jer mi se činilo jako zanimljivim za korišćenje i omogućavao je jednostavnije prebacivanje iz *.ipynb* formata u *pdf*, koji bi takođe imao i priložen korišćen kod. Sam jezik Python je takođe uticao na izbor ovog okruženja, zbog jednostavnosti pisanja koda i ogromnog broja modula za istraživanje skupa podataka.
# 

# In[212]:


#osnovni moduli za korišćenje

from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ###  Skup podataka *employee_reviews* <a class="anchor" id="df"></a>
# 
# Naš skup podataka je preuzet sa <a href="https://www.kaggle.com/petersunga/google-amazon-facebook-employee-reviews">linka</a>. Skup podataka se sastoji od zaposlenih ljudi(bivših i sadašnjih) i njihovih recenzija i ocena radnog mesta. 

# In[213]:


df = pd.read_csv('./employee_reviews.csv')
df.head(5) #razlog komentarisanja je jako los ispis 


# ### Osnovna svojstva <a class='anchor' id='features'></a>
# Kao što se može videti iz priliženog, radi se o poprilično velikom skupu podataka(67529 kolona i 17 atributa). Isto tako, iz priloženih podataka dole, možemo videti o kakvim tipovima je reč. Takođe, da određeni atributi imaju *null* vrednosti("summary" i "advice-to-mgmt"). Daljim pregledom našeg skupa podataka smo naišli na *"none"* vrednosti, koje samo pokazuju da korisnici kad su unosili recenzije o zadatoj firmi su izostavili neke stvari, koje su verovatno smatrali neobaveznim za unošenje. Na nama je da odlučimo kako ćemo ukloniti takve vrednosti, dva poznata načina su: uklanjanje svih redova u kojima se pojavljuju neke od tih *neželjenih* vrednosti ili postavljanje tih nepostojećih na neku podrazumevanu vrednost. Možda uklanjanje svih tih *neželjenih* vrednosti, za primer našeg skupa uzećemo tu vrednost *"none"*, možda i nije toliko dobar pristup(ranijim testiranjem se ispostavilo da ukloni više od 2/3 redova), jer možda gubimo bitne informacije u nekim drugim atributima koje su nam bitni za dalju analizu. Resenje koje bi možda bilo pametnije da je uklonimo samo za nama bitne kolone.

# In[214]:


#veličina našeg skupa
df.shape


# In[215]:


#osnovna svojstva kao što su tipovi atributa i neke statističke ocene
print("Info: ", df.info())
print("Osnovna: ", df.describe())


# In[216]:


#brojimo null vrednosti da znamo da ih uklonimo
df.isnull().sum()


# In[217]:


#izvlacimo svojstva, koja ce nam verovatno biti potrebna
features = df.columns
features


# Kao što smo gore naveli, *"none"* vrednost je verovatno neka podrazumevana vrednost za prazno polje, tako da je možda bolje čak i ne uklanjati u svim atributima, sem u nekim bitnijim atributima.

# In[218]:


# iz ovog vidimo da ako uklonimo sve none vrednosti koje imamo, 
# mozemo da izgubimo dosta podataka
for feature in features:
    print("feature name: %s | count `none` values: %d" 
          %(feature, df[df[feature] == 'none'].shape[0]))


# Ovde već vidimo da je jedan od kategoričkih atributa za koje sam mislio da 
# će nam biti korisniji za dalju analizu, *advice-to-mgmt*, jako loš, budući da dosta zaposlenih nije htelo da napiše neki tekst u tom polju. Uklanjanjem none vrednosti na osnovu tog atributa bi dovelo do brisanja oko 30000 redova, što je već pola skupa. Pored toga, bez obzira što nisam razmatrao o njoj, kolona *location* ima isto 25085 *"none"* polja, i primena klasterovanja ovde se isto ne isplati(doduše ne bi se isplatila jer je početni zadatak bio klasterovanje teksta, a ne reči).

# In[219]:


#da iskopamo te srecnike koji nemaju konstruktivne kritike uopste :D
df[df["cons"] == 'none']


# ## Preprocesiranje. Istraživačka analiza podataka - EDA. <a class="anchor" id="preproc"></a>
# 
# Možda nas zanima koliko je recenzija za svaku firmu postavljeno, čisto da znamo kojih ima najviše, a kojih najmanje.

# In[220]:


#cisto radi vezbe i zabave, grupisacemo da vidimo koliko ima zaposlenih u kojoj firmi
group_by_company = df.groupby('company').size()
group_by_company


# Kao što vidimo, najviše recenzija ima za *Amazon* i *Microsoft*, dok najmanje za *Netflix* i *Facebook*. 
# Takodje pošto smo prilikom gledanja našeg skupa zaposlenih uvideli da postoje sadašnji i bivši zaposleni, možda ne bi bilo loše da odnos bivših i sadašnjih radnika prikažemo.

# In[221]:


#isto tako smo mogli da grupisemo po poslu
group_by_job_title = df.groupby('job-title').size()
#odavde me zanima recimo koliko ima trenutnih, koliko ima bivsih zaposlenih
employee_ratio = df['job-title']
current_former = employee_ratio.map(lambda x : 'former' if 'Former' in x else 'current')
print(current_former.value_counts())
sns.countplot(current_former)
plt.show()


# In[222]:


#i po lokaciji
group_by_location = df.groupby('location').size()


# In[223]:


#a i po datumu
group_by_date = df.groupby('dates').size()
max_date = max(group_by_date.values)
maybe_important_date = list(filter(lambda x : x[1] == max_date, list(group_by_date.items())))
maybe_important_date_name = maybe_important_date[0][0]
maybe_important_date_date = maybe_important_date[0][1]
print("dan kad je najvise komentara okaceno %s : %s\n" 
       % (str(maybe_important_date_name), maybe_important_date_date))
#iz ovoga vidimo da je na taj dan najvise napisano izvestaja o amazonu
df[df['dates'] == maybe_important_date_name].groupby('company').size()


# Procenat podataka na osnovu kompanije će nam reći zastupljenost recenzija za svaku datu kompaniju
# u našem skupu podataka. Ovaj podatak možemo iskoristiti da one manje zastupljene(kad smo ih
# gore prebrojavali, videli smo da se radi o *Facebooku* i *Netflixu*) izbacimo.

# In[224]:


#preprocesiranje podataka
#procenat podataka na osnovu kompanije
df["company"].value_counts()/len(df["company"])*100


# In[225]:


df = df.drop(['Unnamed: 0', 'link'], axis=1)
df = df.loc[~df['company'].isin(['facebook', 'netflix']), :]
df = df.loc[df['dates'] != 'none', :]
df['year'] = pd.to_datetime(df['dates'], errors='coerce').dt.year
df.dropna(how='all', inplace=True)
df.dropna(subset=['company', 'summary', 'year', "overall-ratings", "job-title", "advice-to-mgmt"], inplace=True)
df['year'] = df['year'].astype('int64')
df.head()


# In[226]:


df.shape


# In[227]:


df.groupby('company')['overall-ratings'].describe()


# ##### Broj recenzija za svaku kompaniju <a class="anchor" id="eda1"></a>
# 

# In[228]:


rw_count = df['company'].value_counts().sort_values(ascending=True)
cmp_labels = rw_count.index.tolist()
cmp_index = np.arange(len(cmp_labels))
sns.set(style='whitegrid')
plt.figure(figsize=(12,9))
sns.barplot(cmp_index, rw_count)
plt.xticks(cmp_index, cmp_labels)
plt.xlabel('Kompanije')
plt.ylabel('Broj recenzija')
plt.title('Raspodela recenzija po kompanijama')
plt.show()


# Zaključak koji izvlačimo iz ovoga je da Amazon ima najviše recenzija, dok Google najmanje.

# #### Broj recenzija po godinama <a class="anchor" id="eda2"></a>

# In[229]:


yr_count = df['year'].value_counts().sort_index(ascending=True)
yr_labels = yr_count.index.tolist()
yr_index = np.arange(len(yr_labels))

#vecina recenzija je napisana u prethodnih 3-4 godine
plt.figure(figsize=(12,9))
sns.barplot(yr_index, yr_count)
plt.xticks(yr_index, yr_labels)
plt.xlabel('Godina')
plt.ylabel('Broj recenzija')
plt.title('Godisnji broj recenzija')
plt.show()


# Zaključak koji izvlačimo ovde je da u poslednjih četiri ili pet godina najviše recenzija napisano(potencijalno možemo da eliminišemo recenzije iz godina 2008-2013, ali pre toga moramo da proverimo da li se iz godišnjeg broja recenzija po firmama može izvesti isti zaključak).

# In[230]:


fig, axs = plt.subplots(2,2,figsize=(12,9))
companies = [['microsoft', 'apple'], ['google', 'amazon']]

for i in range(2):
    for j in range(2):
        company = companies[i][j]
        yr_count = df[df['company'] == company]['year'].value_counts().sort_index(ascending=True)
        yr_labels = yr_count.index.tolist()
        yr_index = np.arange(len(yr_labels))
        g = sns.barplot(yr_index, yr_count, tick_label = yr_labels, ax=axs[i][j])
        g.set(xticklabels=yr_labels)
        axs[i][j].set_xlabel('Godine')
        axs[i][j].set_ylabel('Broj recenzija')
        axs[i][j].set_title("%s: Godisnji broj recenzija" % (company.title()))
plt.tight_layout()


# Zaključak koji izvlačimo ovde je da je najvise recenzija bilo u poslednjih četiri-pet godina.

# In[231]:


df = df[df['year'].isin(["2018", "2017", "2016", "2015"])]


# ##### Prikaz promene prosečne ocene kompanije od 2015. do 2018. godine <a class="anchor" id="eda3"></a>

# In[232]:


fig, ax = plt.subplots(figsize=(15,9))
df.groupby(['year', 'company'])['overall-ratings'].mean().unstack().plot(ax=ax)
plt.xlabel("Godine")
plt.ylabel("Uopstene recenzije")
plt.title("Godisnje uopstene recenzije")
plt.show()


# Možemo primetiti da u svakoj kompaniji, sem u *Apple-u*, posle 2015. raste prosečna ocena

# In[233]:


current_employee_count = df['job-title'].str.split('-', expand=True)[0].value_counts().sort_values(ascending=True)
employee_labels = current_employee_count.index.tolist()
employee_index = np.arange(len(employee_labels))


plt.figure(figsize=(12, 9))
sns.barplot(employee_index, current_employee_count)
plt.xticks(employee_index, employee_labels)

plt.xlabel('Tip zaposlenih')
plt.ylabel('Broj recenzija')
plt.title('Raspodela recenzija na osnovu zaposlenog(Bivsi radnik vs Sadasnji radnik)')
plt.show()


# ##### Prosečna ocena svih kompanija za svaki kriterijum <a class="anchor" id="eda4"></a>

# In[234]:


ratings_comp = df[["company", "work-balance-stars", "culture-values-stars", "carrer-opportunities-stars", "comp-benefit-stars", "senior-mangemnet-stars"]]
ratings_comp.set_index(["company"], inplace=True)
ratings_comp = ratings_comp[~(ratings_comp[["work-balance-stars", "culture-values-stars", "carrer-opportunities-stars", "comp-benefit-stars", "senior-mangemnet-stars"]] == "none").any(axis=1)]
ratings_comp[["work-balance-stars", "culture-values-stars", "carrer-opportunities-stars", "comp-benefit-stars", "senior-mangemnet-stars"]] = ratings_comp[["work-balance-stars", "culture-values-stars", "carrer-opportunities-stars", "comp-benefit-stars", "senior-mangemnet-stars"]].apply(pd.to_numeric)
group = ratings_comp.groupby("company")["work-balance-stars", "culture-values-stars", "carrer-opportunities-stars", "comp-benefit-stars", "senior-mangemnet-stars"].mean()
group.columns = ["Work Balance", "Culture Values", "Career Opportunities", "Company Benefits", "Senior Management"]
group = group.transpose()
group


# In[235]:


def rating_per_company(rating_type, title, color):
    work_ratings = df[['company', rating_type]]
    work_ratings = work_ratings[~(work_ratings[[rating_type]] == "none").any(axis=1)]
    work_ratings[rating_type] = work_ratings[rating_type].apply(pd.to_numeric)
    group = work_ratings.groupby(work_ratings["company"].str.title())[rating_type].mean().reset_index()
    group.sort_values([rating_type],inplace=True)
    group.set_index('company').plot.barh(legend=False, figsize=(12, 10), color=color)
    plt.title('{} Rating'.format(title))
    plt.xlabel('Rating')
    plt.ylabel('Companies')


# In[236]:


rating_per_company('work-balance-stars', "Work Balance", "r")


# In[237]:


rating_per_company("culture-values-stars", "Culture Values", "b")


# In[238]:


rating_per_company("carrer-opportunities-stars", "Career Opportunities", "g")


# In[239]:


rating_per_company("comp-benefit-stars", "Company Benefits", "#cdca04")


# In[240]:


rating_per_company("senior-mangemnet-stars", "Senior Management", "#e06743")


# In[241]:


fig, axs = plt.subplots(2,2, figsize=(12, 9), facecolor='w', edgecolor='k')
companies = ['microsoft', 'apple', 'google', 'amazon']
axs = axs.ravel()


for i, company in enumerate(companies):
    current_employee_count = df.loc[df['company'] == company]['job-title'].str.split('-', expand=True)[0].value_counts().sort_values(ascending=True)
    employee_labels = current_employee_count.index.tolist()
    employee_index = np.arange(len(employee_labels))
        
    bars = axs[i].bar(employee_index, current_employee_count, tick_label=employee_labels)
    bars[0].set_color('gray')
    bars[1].set_color('b')
    axs[i].set_xlabel('Tip zaposlenih')
    axs[i].set_ylabel('Broj recenzija')
    axs[i].set_title('%s: Recenzije (Bivsi zaposleni vs Sadasnji)' %(company.title()))
    
fig.tight_layout()


# Zaključak je da najviše napisanih recenzija za svaku kompaniju dolazi od trenutno zaposlenih ljudi.

# In[242]:


df['job-title'].str.split(' - ', expand=True)[1].value_counts().head(5)


# #### Prosečna ocena za kompaniju na osnovu bivšeg i sadašnjeg zaposlenog <a class="anchor" id="eda5"></a>

# In[243]:


fig, axs = plt.subplots(2,2, figsize=(12, 9), facecolor='w', edgecolor='k')
companies = ['microsoft', 'apple', 'google', 'amazon']
axs = axs.ravel()

for i, company in enumerate(companies):
    
        job_rating = df[df['company'] == company][['job-title', 'overall-ratings']]
        job_rating['job-title'] = job_rating['job-title'].str.split(' - ', expand=True)[0]
        job_rating_count = job_rating.groupby('job-title')['overall-ratings'].mean().sort_values(ascending=True)

        bars = axs[i].bar([0,1], job_rating_count, tick_label = ['Current Employee', 'Former Employee'])
        bars[0].set_color('gray')
        bars[1].set_color('b')
        axs[i].set_title('{}: Prosecna recenzija na osnovu zaposlenog (Sadasnji vs Bivsi)'.format(company.title()))
        axs[i].set_xlabel("Tip Zaposlenog")
        axs[i].set_ylabel("Prosecna recenzija")

fig.tight_layout()


# Zaključak je da *Amazon* ima najveću razliku u prosečnoj ocenu koje su dali bivši i sadašnji zaposleni.

# ### Izbor korisnijih kolona za dalju analizu <a class="anchor" id="preproc1"></a>
# <hr />
# 
# Od svih kolona, deluje mi da su nam samo *summary*, *pros*, *cons*, *overall-ratings* korisnije, jer iz njih možemo tekst da izvučemo da bi sproveli dalju analizu, dok prosečnu ocenu možemo da pretvorimo u kategorički atribut. Daljom analizom *summary* kolone, može se videti da je većina tih sažetaka jako kratka i da sadrži ili par reči ili poziciju zaposlenog. 

# In[244]:


new_df = df.iloc[:,[0,4,5,6,8]]
new_df.head()


# In[245]:


review = new_df['overall-ratings']
ctgr = review.map(lambda x : 'positive' if x > 3 else 'negative')
new_df['overall-ratings'] = ctgr
new_df.head()


# In[246]:



new_df['overall-ratings'].value_counts()


# ## Klasterovanje <a class="anchor" id="clust1"></a>
# <hr />
# 
# Klasterovanje ili klaster analiza je pronalaženje grupa objekata takvih da su objekti u grupi međusobno slični, odnosno da su objekti u različitim grupama međusobno različiti.
# 
# 

# ## Klasterovanje teksta <a class="anchor" id="clust2"></a>
# <hr />
# 
# Pošto računari razumeju samo brojeve, tekst je potrebno pretvoriti u brojeve.
# Klasterovanje teksta ima primenu u automatskoj organizaciji dokumenata(tekstova), izvlačenju tema ili brzih informacija. Svodi se na korišćenje *deskriptora*, koji su skupovi reči koji opisuju sadržaj klastera. Cilj klasterovanja
# teksta je da grupiše tekstove u razdvojene skupove klastera.
# 
# Postupak prilikom klasterovanja teksta
# * tokenizacija
# * okrnjavanje(*eng. stemming*)
# * uklanjanje zaustavnih reči i znakova interpukcije
# * računanje učestalosti termova svih tokena(*CountVectorizer, TF-IDF, Word2Vec*)
# * klasterovanje
# * ocenjivanje i grafički prikaz

# Da bi mogli da sprovedemo klasterovanje teksta, koristimo NLTK. On je skup alatki za obradu prirodnih jezika(NLP), koji u svojem sastavu ima i zaustavne reči(*eng. - stopwords*) i alat koji svodi na koren reči(ne nužno morfološki koren). Takođe nije samo razvijen da obrađuje engleski jezik, već može i ostale. Pored ovih, omogućava i druge obrade, vezane za semantičko značenje, označavanje, parsiranje.

# In[247]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
sno = nltk.stem.SnowballStemmer('english')
stop = set(stopwords.words('english'))

def cleanpunc(sentence): 
    cleaned = re.sub(r'[?|!|\'|"|#|<|>]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

#print(stop)
#print(sno.stem("tasting"))


# In[248]:


i=0
stringic=' '
final_string=[]
all_positive_words=[] 
all_negative_words=[] 
s=''
for sent in new_df['pros'].values:
    filtered_sentence=[]
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (new_df['overall-ratings'].values)[i] == 'positive': 
                        all_positive_words.append(s) 
                    if(new_df['overall-ratings'].values)[i] == 'negative':
                        all_negative_words.append(s) 
                else:
                    continue
            else:
                continue 
    stringic = b" ".join(filtered_sentence) 
    
    final_string.append(stringic)
    i+=1


# In[249]:


new_df['CleanedText'] = final_string
new_df['CleanedText'] = new_df['CleanedText'].str.decode('utf-8')


# ## Brojači reči

# ## *CountVectorizer*
# <hr />
# 
# CountVectorizer koristi princip "*džak reči*", što znači da se svaka reč u svakom tekstu broji i predstavlja kao matrica.

# In[250]:


from sklearn.feature_extraction.text import CountVectorizer
c_vect = CountVectorizer()
bow = c_vect.fit_transform(new_df['CleanedText'].values)
bow.shape


# In[251]:


terms = c_vect.get_feature_names()
terms[:20]


# #### "Lakat" metoda
# 
# "Lakat" metoda se koristi u klaster analizi za otkrivanje optimalnog broja klastera za zadati skup podataka. Nekad nije dobro osloniti(jer je višeznačna i ona će odrediti optimalan broj klastera za naš raspon, ali je pitanje da li je raspon broja klastera koji smo uzeli u obzir dobar) se na nju i da je bolje koristiti koeficijent Senke(*eng. Silhouette coefficient*). Za svako k računamo unutrašnju sumu kvadrata klastera(WSS), i iscrtavamo krivu za to zadato k. Mesto gde se nalazi "lakat" se uzima kao optimalan broj klastera za zadat skup podataka.

# In[252]:



def elbow_method(text, num_clus):
    squared_errors = []
    for cluster in num_clus:
        print("Fit %d clusters." %(cluster))
        kmeans = KMeans(n_clusters = cluster).fit(bow) 
        squared_errors.append(kmeans.inertia_)
    
    optimal_clusters = np.argmin(squared_errors) + 2  
    plt.plot(num_clus, squared_errors)
    plt.title("Elbow Curve to find the no. of clusters.")
    plt.xlabel("Number of clusters.")
    plt.ylabel("Squared Loss.")
    xy = (optimal_clusters, min(squared_errors))
    plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')
    plt.show()

    print ("The optimal number of clusters obtained is - ", optimal_clusters)
    print ("The loss for optimal cluster is - ", min(squared_errors))
    
    return optimal_clusters
num_clusters = elbow_method(bow, range(2,12,2))


# In[253]:


model = KMeans(n_clusters = num_clusters,init='k-means++', n_jobs = -1,random_state=99)
model.fit(bow)


# In[254]:


bow_labels = model.labels_
bow_cluster_center=model.cluster_centers_


# Koeficijent senke je unutrašnji kriterijum provere našeg klasterovanja. Uzima vrednosti između -1 i 1. Što je bliži koeficijent jedinici, govorimo o jako dobro razdvojenim klasterima i da su naši parametri za klasterovanje prikladni. Negativna vrednost nam govori o mešavini podataka u klasterima.

# In[255]:


from sklearn import metrics as met
bow_silhouette_score = met.silhouette_score(bow, bow_labels, metric='euclidean')


# In[256]:


bow_silhouette_score


# In[257]:


new_df['Bow Cluster Label'] = model.labels_
new_df.head(2)


# In[258]:


#da vidimo kako su grupisani podaci
new_df.groupby(['Bow Cluster Label'])['pros'].count()


# In[259]:


def top_terms_per_cluster(centers, num_of_clusters, terms, num_of_terms):
    print("Top terms per cluster:")
    order_centroids = centers.argsort()[:, ::-1]
    for i in range(num_of_clusters):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :num_of_terms]:
            print(' %s' % terms[ind], end='')
            print()
top_terms_per_cluster(bow_cluster_center, num_clusters, terms, 10)


# In[260]:


import matplotlib.pyplot as plt
def plot_cluster_points(cluster_label_str, feature, num_clusters):
    plt.bar([x for x in range(num_clusters)], new_df.groupby([cluster_label_str])[feature].count(), alpha = 0.4)
    plt.title('KMeans cluster points')
    plt.xlabel("Cluster number")
    plt.ylabel("Number of points")
    plt.show()

plot_cluster_points('Bow Cluster Label', 'pros', num_clusters)


# In[306]:


def print_review_asigned_to_cluster(cluster_label_str, feature, num_clusters):
    for i in range(num_clusters):
        print("review of assigned to cluster ", i)
        print("-" * 70)
        print(new_df.iloc[new_df.groupby([cluster_label_str]).groups[i][0]][feature])
        print('\n')
        print(new_df.iloc[new_df.groupby([cluster_label_str]).groups[i][5]][feature])
        print('\n')
        print(new_df.iloc[new_df.groupby([cluster_label_str]).groups[i][10]][feature])
        print('\n')
        print(new_df.iloc[new_df.groupby([cluster_label_str]).groups[i][20]][feature])
        print("_" * 70)
    
print_review_asigned_to_cluster('Bow Cluster Label', 'pros', num_clusters)


# In[262]:


import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    
plot_tsne_pca(bow, bow_labels)


# Klasteri 1, 2 i 4 imaju najviše grupisanih recenzija. Da vidimo šta je najčešće:

# In[315]:


print("Cluster 1: ")
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[1][2]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[1][12]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[1][22]]['pros'])
print("\nCluster 2: ")
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[2][2]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[2][12]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[2][22]]['pros'])
print("\nCluster 4: ")
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[4][2]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[4][12]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[4][22]]['pros'])


# Klaster 5 ima najmanje recenzija. Da ispišemo sve, kad ih je toliko malo:

# In[322]:


print("Cluster 5: ")
print("_" * 70)
for i in range(0, 25, 1):
    print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[5][i]]['pros'])
    print("_" * 70)


# ## TF-IDF 
# 
# *TF-IDF* (*eng. Term frequency-inverse document frequency*) je još jedan od načina za prevođenje niski u brojeve. Ocena reči se izračunava proizvodom učestalosti zadate reči i 
# inverznom ferkvencijom(*"što je učestalija u ostalim tekstovima, ocena će biti manja"*).
# 

# In[263]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
tfidf = tfidf_vect.fit_transform(new_df['CleanedText'].values)
tfidf.shape


# In[264]:


num_clusters = elbow_method(tfidf, range(2,12,2))


# In[265]:


model_tf = KMeans(n_clusters = num_clusters)
model_tf.fit(tfidf)


# In[266]:


labels_tf = model_tf.labels_
cluster_center_tf=model_tf.cluster_centers_


# In[267]:


terms1 = tfidf_vect.get_feature_names()
terms1[:10]


# In[268]:


silhouette_score_tf = met.silhouette_score(tfidf, labels_tf, metric='euclidean')
silhouette_score_tf


# In[269]:


# Giving Labels/assigning a cluster to each point/text 
tfidf_df = new_df
tfidf_df['Tfidf Cluster Label'] = model_tf.labels_
tfidf_df.head(5)


# In[270]:


tfidf_df.groupby(['Tfidf Cluster Label'])['pros'].count()


# In[271]:


top_terms_per_cluster(cluster_center_tf, num_clusters, terms1, 10)


# In[272]:


plot_cluster_points('Tfidf Cluster Label', 'pros', num_clusters)


# In[273]:


print_review_asigned_to_cluster('Tfidf Cluster Label', 'pros', num_clusters)


# In[274]:


plot_tsne_pca(tfidf, labels_tf)


# Klasteri 2, 4 i 5 imaju najviše recenzija. Vidimo da se u 5 pojavljuju i o malo "negativnijim" stranama posla, kao što su smene od 10 sati.

# In[324]:


print("Cluster 2: ")
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[2][2]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[2][12]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[2][22]]['pros'])
print("\nCluster 4: ")
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[4][2]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[4][12]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[4][22]]['pros'])
print("\nCluster 5: ")
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[5][2]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[5][12]]['pros'])
print("_" * 70)
print(new_df.iloc[new_df.groupby(['Bow Cluster Label']).groups[5][22]]['pros'])


# ### *Word2Vec*
# <hr />
# 
# *Reč u vektor* (*eng. Word2Vec*) je još jedan poznat algoritam da pretvorimo reči u brojeve(*eng. - word embedings*). Svaka reč dobije određen niz brojeva(rezultat pretvaranja je ogromna "matrica"), i pošto imamo nizove brojeva, moći ćemo rastojanjem izmedju vektora da utvrdimo sličnost između reči. 
# 

# In[304]:


i=0
list_of_sent=[]
for sent in new_df['CleanedText'].values:
    list_of_sent.append(sent.split())

#list_of_sent[:10]


# In[303]:


i=0
list_of_sent_train=[]
for sent in new_df['CleanedText'].values:
    filtered_sentence=[]
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_of_sent_train.append(filtered_sentence)
#list_of_sent_train[:10]


# In[277]:


import gensim
w2v_model=gensim.models.Word2Vec(list_of_sent_train,size=100, workers=4)


# In[278]:


sent_vectors = []; 
for sent in list_of_sent_train: 
    sent_vec = np.zeros(100) 
    cnt_words =0; 
    for word in sent: 
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
sent_vectors = np.array(sent_vectors)
sent_vectors = np.nan_to_num(sent_vectors)
sent_vectors.shape


# In[279]:


num_clusters = elbow_method(sent_vectors, range(2,12,2))


# In[280]:


model_w2v = KMeans(n_clusters = num_clusters)
model_w2v.fit(sent_vectors)


# In[281]:


word_cluster_pred=model_w2v.predict(sent_vectors)
word_cluster_pred_2=model_w2v.labels_
word_cluster_center=model_w2v.cluster_centers_


# In[282]:


silhouette_score_tf = met.silhouette_score(sent_vectors, word_cluster_pred_2, metric='euclidean')
silhouette_score_tf


# In[283]:


df_w2v = tfidf_df
df_w2v['AVG-W2V Cluster Label'] = model_w2v.labels_
df_w2v.head()


# In[284]:


# How many points belong to each cluster ->
df_w2v.groupby(['AVG-W2V Cluster Label'])['pros'].count()


# In[285]:


plot_cluster_points('AVG-W2V Cluster Label', 'pros', num_clusters)


# In[307]:


print_review_asigned_to_cluster('AVG-W2V Cluster Label', 'pros', num_clusters)


# In[287]:


#ne znam koliko je ispravno ovo
import scipy
csr_sent_vectors = scipy.sparse.csr_matrix(sent_vectors)
plot_tsne_pca(csr_sent_vectors, word_cluster_pred_2)


# 
# 

# ## Klasterovanje putem algoritma DBSCAN
# <hr />
# 

# In[288]:


dbscan_df = df_w2v.head(10000)
dbscan_df.head()


# In[289]:


#print(dbscan_df.shape)
i=0
list_of_sent=[]
for sent in dbscan_df['CleanedText'].values:
    list_of_sent.append(sent.split())

    
#list_of_sent[:10]


# In[290]:


i=0
list_of_sent_train=[]
for sent in dbscan_df['CleanedText'].values:
    filtered_sentence=[]
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_of_sent_train.append(filtered_sentence)
#list_of_sent_train[:10]


# In[291]:


w2v_model=gensim.models.Word2Vec(list_of_sent_train,size=100, workers=4)


# In[292]:


db_sent_vectors = []; 
for sent in list_of_sent_train: 
    sent_vec = np.zeros(100) 
    cnt_words =0; 
    for word in sent: 
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    db_sent_vectors.append(sent_vec)
db_sent_vectors = np.array(db_sent_vectors)
db_sent_vectors = np.nan_to_num(db_sent_vectors)
db_sent_vectors.shape


# In[293]:


from sklearn.cluster import DBSCAN


dbscan_model = DBSCAN(eps = 0.8, min_samples = 2, n_jobs=-1, metric='euclidean')
dbscan_model.fit(db_sent_vectors)


# In[294]:


dbscan_df['AVG-W2V DBSCAN Cluster Label'] = dbscan_model.labels_
dbscan_df.head(2)


# In[295]:


db_word_cluster_pred_2=dbscan_model.labels_

dbscan_df.groupby(['AVG-W2V DBSCAN Cluster Label'])['pros'].count()


# Već za $eps=0.8$, DBSCAN algoritam sve tačke grupiše u jedan klaster. Tako da ovde ne možemo izvući nikakve interesnatne teme vezane za naše recenzije.

# ## Hiearhijsko klasterovanje

# In[296]:


from scipy.cluster import hierarchy
plt.figure(figsize=(14,14))
dendro=hierarchy.dendrogram(hierarchy.linkage(db_sent_vectors,method='ward'), orientation="right")
plt.show()


# In[297]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
agg=cluster.fit_predict(db_sent_vectors)


# In[298]:


agg_df = dbscan_df
agg_df['AVG-W2V AGG Cluster Label'] = cluster.labels_
agg_df.head()


# In[299]:


silhouette_score_agg = met.silhouette_score(db_sent_vectors, cluster.labels_, metric='euclidean')
silhouette_score_agg


# In[300]:


agg_df.groupby(['AVG-W2V AGG Cluster Label'])['pros'].count()


# In[301]:


plt.bar([x for x in range(2)], agg_df.groupby(['AVG-W2V AGG Cluster Label'])['pros'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()


# In[308]:


for i in range(2):
    print("reviews of assigned to cluster ", i)
    print("-" * 70)
    print(agg_df.iloc[agg_df.groupby(['AVG-W2V AGG Cluster Label']).groups[i][0]]['pros'])
    print('\n')
    print(agg_df.iloc[agg_df.groupby(['AVG-W2V AGG Cluster Label']).groups[i][5]]['pros'])
    print('\n')
    print(agg_df.iloc[agg_df.groupby(['AVG-W2V AGG Cluster Label']).groups[i][10]]['pros'])
    print('\n')
    print(agg_df.iloc[agg_df.groupby(['AVG-W2V AGG Cluster Label']).groups[i][20]]['pros'])

    print("_" * 70)


# ## Finalni zaključak
# 
# "Lakat" metoda je za K-sredina algoritam pokazala da je optimalan broj klastera 6, dok koeficijent senke za sva tri slučaja korišćenja algoritma ne prelazi *0.1*. 
# U slučaju *tf-idf-a*, najveći broj recenzija je grupisao za jedan klaster(*Cluster 4: 27304*), verovatno zbog toga što su reči postale jako učestale u skupu tekstova(pošto smo radili sa kolonom *'pros'*, verovatno su se reči kao što su 'posao', 'benefit', 'zabava', 'hrana' dosta pojavljivale)
# U slučaju *bow-a*, zabeležen je najmanji broj recenzija(*Cluster 5: 25*), dok su ostale recenzije raspodeljene jednako.
# U slučaju *word2vec-a*, se trudi da raspodeli recenzije ravnomerno po klasterima, i za zadate parametre klasterovanja daje najbolji koeficijent senke za K-sredina klasterovanje.
# 
# 
# Zbog bržeg izvršavanja, za *DBSCAN* i Hierarhijsko smo uzeli samo 10000 podataka iz čitavog skupa.
# DBSCAN je jako loše odradio svoj posao na skupu od 10000 podataka, grupisajući sve recenzije u jedan klaster(verovatno bi mogla bolja raspodela da se postigne smanjivanjem eps, jer je za 0.2 delio recenzije na dva klastera(doduše Klaster 1 je imao 12 recenzija samo.)).
# Za hierarhijsko klasterovanje smo koristili *avg word2vec* metodu, zato što raspodeljuje klastere jednako, i imala je najveću ocenu senke(približno 0.3).

# ## Literatura <a class="anchor" id="lit"></a>
# <hr />
# 
# Literatura i kodovi koji su mi pomogli za projekat se mogu naći na ovim linkovima:
#  - <a href="https://github.com/avourakis/Employee-Reviews-Analysis">eda i preprocesiranje skupa podataka</a>
#  - <a href="https://www.kaggle.com/karthik3890/text-clustering">klasterovanje</a>
#  - <a href="https://www.youtube.com/watch?v=FrmrHyOSyhE">countvectorizer</a>
#  - <a href="https://www.youtube.com/watch?v=hXNbFNCgPfY">tfidf</a>
#  - <a href="https://www.youtube.com/watch?v=LSS_bos_TPI">word2vec</a>
#  - <a href="https://gist.github.com/aparrish/2f562e3737544cf29aaf1af30362f469">word2vec</a>
#  - slajdovi i kodovi sa predavanja i vežbi

# In[ ]:





from urllib.request import urlopen
from numpy import genfromtxt, zeros, linspace, matrix, corrcoef, arange
from numpy.random import rand
from pylab import plot, show, figure, subplot, pcolor, colorbar, xticks, yticks
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import networkx
from networkx import find_cliques


#Data Importing and Visulization
u = urlopen('http://aima.cs.berkeley.edu/data/iris.csv').read()

localFile = open('iris.csv', 'wb')
localFile.write(u)
localFile.close()

data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))               #read the first 4 columns   
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)         #read the fifth column
set = (['setosa', 'versicolor', 'virginica'])                               #Build a collection of unique elements


plot(data[target=='setosa',0], data[target=='setosa',2],'bo')               #plotting the values of sepal length vs sepal width for each class
plot(data[target=='versicolor',0], data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0], data[target=='virginica',2],'go')    
show()

#Classification
t = zeros(len(target))
t[target == 'sertosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3

classifier = GaussianNB()
classifier.fit(data,t)                  #Training on the Iris dataset
#print(classifier.predict(data[0]))

train, test, t_train, t_test = train_test_split(data, t, test_size=0.4, random_state=0)
classifier.fit(train,t_train)               #Training the classifier
print(classifier.score(test, t_test))       #Testing the classifier

#Clustering
kmeans = KMeans(n_clusters=3, init='random')         #Initialization
kmeans.fit(data)                             #Executing the kmeans model

c = kmeans.predict(data)

# figure()
# subplot(211)                                    #Top figure with real classes
# plot(data[t==1,0],data[t==1,2],'bo')
# plot(data[t==2,0],data[t==2,2],'ro')
# plot(data[t==3,0],data[t==3,2],'go')
# subplot(212)                                    #Bottom figure with classes assigned automatically
# plot(data[c==1,0],data[t==1,2],'bo',alpha=.7)
# plot(data[c==2,0],data[t==2,2],'go',alpha=.7)
# plot(data[c==0,0],data[t==0,2],'mo',alpha=.7)
# show()

#Regression
x = rand(40,1)                                      #Explanatory variable
y = x*x*x+rand(40,1)/5                              #Dependent variable

linReg = LinearRegression()
linReg.fit(x,y)

xx = linspace(0,1,40)
plot(x,y,'o', xx, linReg.predict(matrix(xx).T), '--r')
show()
print(mean_squared_error(linReg.predict(x),y))

#Correlation
corr = corrcoef(data.T)
print(corr)

pcolor(corr)
colorbar()
xticks(arange(0.5,4.5),['sepal length',  'sepal width', 'petal length', 'petal width'],rotation=-20)
yticks(arange(0.5,4.5),['sepal length',  'sepal width', 'petal length', 'petal width'],rotation=-20)
show()

#Dimensionality Reduction
pca = PCA(n_components=2)               #Principle Component Analysis
pcad = pca.fit_transform(data)
plot(pcad[target=='setosa',0],pcad[target=='setosa',1],'bo')
plot(pcad[target=='versicolor',0],pcad[target=='versicolor',1],'ro')
plot(pcad[target=='virginica',0],pcad[target=='virginica',1],'go')
show()

print(pca.explained_variance_ratio_)        #Determine how much information is stored in the PC's by looking at the variance ratios
print(1-sum(pca.explained_variance_ratio_)) #This shows how much information we lost durin the transformation process

data_inv = pca.inverse_transform(pcad)      #Apply the inverse transformation to get the original data back

#Networks Mining
G = networkx.read_gml('lesmiserables.gml')
networkx.draw(G,node_size=0,edge_color='b',alpha=.2,font_size=7)

Gt = G.copy()
dn = networkx.degree(Gt)
for n in Gt.nodes():
    if dn[n] <= 10:
        Gt.remove_node(n)

networkx.draw(Gt, node_size=0, edge_color = 'b', alpha=.2, font_size=12)
cliques = list(find_cliques(G))
print(max(cliques, key=lambda l: len(1)))








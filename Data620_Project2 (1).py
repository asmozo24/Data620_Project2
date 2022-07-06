#!/usr/bin/env python
# coding: utf-8

# In[ ]:


''''
Project 2

Identify a large 2-node network dataset—you can start with a dataset in a repository.  
Your data should meet the criteria that it consists of ties between and not within two (or more) distinct groups.
Reduce the size of the network using a method such as the island method described in chapter 4 of social network analysis.
What can you infer about each of the distinct groups?

You may work in a small group on the project.
Your code and analysis should be delivered in an IPython Notebook by end of day Sunday. 
''''


# In[ ]:


''''
Some defintiions:
    
A two-mode network refers to a network where the nodes are classified into two distinct types,
and edges can only exist between nodes of different types. In other words, a two-mode network is
a bipartite graph. 

A bipartite graph is a graph decomposed in 02 disjoint sets (domains) with 
no adjacent elements within each domain(meaning, elements of each domain 
does communicate between themselve). example of f(x) = y

In analysis of two-mode networks, one important objective is to explore the relationship between
responses of two types of nodes.


Binary Two-mode Networks:1 or 0

Projection:
Projection is done by selecting one of the sets of nodes and linking two nodes from that set if they were 
connected to the same node (of the other kind).

Unweighted Two-mode Networks...a bipartite graph with no intensity on edge.
Weighted Two-mode Networks...a bipartite graph with intensity of the edge is given.


# In[ ]:


The world's best footballers: the top 100 list

DATASET: World's best top 100 footballers
Link ---> https://www.theguardian.com/football/datablog/2012/dec/24/world-best-footballers-top-100-list#data


Description: csv file, 22×22 matrix, symmetric, binary.
    
Background:
    
The Guardian's choice of the world's top 100 footballers has been unveiled today with Lionel Messi topping the list at No1.
An 11-strong international panel of experts were asked by Guardian Sport to name their top 30 players in action today and 
rank them in order of preference. Players were then scored on their ranking by each panellist: a No1 choice allocated 30pts, 
No2 29pts and so on down to selection No30, given one point.

selection criteria: https://www.theguardian.com/football/blog/2012/dec/20/guardian-world-top-100-footballers
        


# In[77]:


import pandas as pd
import numpy as np
import csv
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import networkx.algorithms.bipartite as bipartite


# In[7]:


# Import the data from github
url = 'https://raw.githubusercontent.com/asmozo24/Data620_Project2/main/World_best_top100_footballers.csv'
top100 = pd.read_csv(url, index_col=0)
print(top100.head(8))


# In[11]:


#Checking any missing value per column
top100.isna().any()

# let's trim the df to only columns we want
df = top100[['Name','Club']]
df

#Renaming
# rename specific column....df.rename(columns = {'old_col1':'new_col1', 'old_col2':'new_col2'}, inplace = True)
# rename all columns....df.columns = ['new_col1', 'new_col2', 'new_col3', 'new_col4']
df.columns = ['Names','Clubs']
# Reduce size of the dataset
#df.iloc[:100]
#top100 = df.iloc[:100, :]
#df


# In[15]:


#frequency indicating how often each of the categories are observed together (crosstab)
crosstab = pd.crosstab([df["Names"]], df["Clubs"])
crosstab


# In[22]:


#Since the ranking was done in 2013, a lot payers usually have 2 to 3 years contract...so I am going to update the matrix
# 1 means the player had played for the club, 0 means the players never played for the club.

crosstab.to_csv('top100Footballers.csv', index = True) #C:\\Users\\owner\\Downloads\\top100Footballers.csv


# In[43]:


# Import the data from github
url1 = 'https://raw.githubusercontent.com/asmozo24/Data620_Project2/main/top50footballers2013.csv'
top50 = pd.read_csv(url1)
top50.shape
#print(top50.head(5))
#print(top50.columns.tolist())
#print(top50.columns.values)
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(top50)


# In[44]:


#Create a network...Let's extract the two node
clubs = list(top50)[1:]
print(clubs)


# In[ ]:


players = list(top50['Names'])
print(players)


# In[49]:


#Let's extract the matrix
top50a = top50.drop(top50.columns[[0]], axis=1)
#top50a


# In[50]:


# Adjacent matrix
adj_matrix = top50a.to_numpy()
print(adj_matrix)


# In[51]:


adj_matrix.shape


# In[58]:


# Network
def create_network(top, bottom, adjacent):
    G = nx.Graph()
    G.add_nodes_from(top,bipartite=0)
    G.add_nodes_from(bottom, bipartite=1)
    for i in range(len(bottom)):
        for j in range(len(top)):
            if adjacent[i,j]==1:
                G.add_edge(bottom[i], top[j])
                
    return G


# In[69]:


#let's call our function to work
G = create_network(clubs,players,adj_matrix)


# In[154]:


# Visualizing the network

fig1 = plt.figure(1, figsize=(8,6), dpi=400)
pos = nx.spring_layout(G)
colors = {0:'r', 1:'b'}
#nodes
nx.draw_networkx_nodes(G, pos=pos, node_size=[G.degree[node]*2 for node in G], alpha = 0.8, node_color=[colors[G.nodes[node]['bipartite']]for node in G])
#edges
nx.draw_networkx_edges(G, pos=pos, alpha=0.4, edge_color='gray')
plt.axis('off')
plt.show()


# In[155]:


#Analysis
# what are the degree of each node..degreeView
#G.degree()
# to get the degree of each node as a list
#degrees = [val for (node, val) in G.degree()]
#print(degrees)
#The node degree is the number of edges adjacent to the node. 
#The weighted node degree is the sum of the edge weights for edges incident to that node.
clubs_pop, players_contract = bipartite.degrees(G, players)
print(clubs_pop)


# In[148]:


#let's see professional contracts in 2013
print(players_contract)


# In[82]:


# how to find the distribution for each club in the top50 2013
# how many time each degree appear
clubs_pop_distribution = Counter(sorted(dict(clubs_pop).values()))
players_contract_distribution = Counter(sorted(dict(players_contract).values()))
print(clubs_pop_distribution)
#There are 13 clubs having at least one contract with a player in the top50
#there are 10 clubs having only 02 contracts with a player in the top50
#There are 06 clubs having 0 contracts with players from the the top50...something is wrong
#...
#there is one clubs having 5 contracts with players in the top50
#there are 02 clubs having 10 contracts with players in the top50


# In[83]:


print(players_contract_distribution)
# There are 14 players from the top50 having only 2 contracts or had played for 02 clubs only in 2013
#There is one player who had only 06 contract or had played for 06 club only in 2013...>maybe sign of instability
#There are 09 players who had only 01 contract or had played for 01 club up till 2013...>son of the club


# In[100]:


# Let plot degree distribution , node1=node 1 distribution, node2 = node 2 distribution
def plot_degree_distr(node1, node2):
    
    #setup plot dimension
    fig = plt.figure(figsize = (11,5), dpi=600 )
    ax = [fig.add_subplot(1,2,i+1) for i in range(2)]
    
    #plot
    ax[0].plot(list(node1.keys()),list(node1.values()), 'bo', linestyle = '-', label='Clubs popularity')
    ax[1].plot(list(node2.keys()),list(node2.values()), 'bo', linestyle = '-.', label = 'Players contracts')
    
    #setup axis
    for axis in ax:
        axis.set_ylabel('Frequency', fontsize=16)
        axis.tick_params(axis = 'both', which = 'major', labelsize = 16)
        axis.legend(loc='upper right', fontsize=12, ncol = 1, frameon = True)
    ax[0].set_xlabel('Degree-Clubs', fontsize = 16)
    ax[1].set_xlabel('Degree-Players', fontsize =16)
    
    plt.tight_layout()
    plt.show()
        


# In[101]:


# let's call the distribution function
plot_degree_distr(clubs_pop_distribution, players_contract_distribution)
# on the club popularity...frequency represent number of clubs for players on x-axis
# there are 13 clubs ..
# there is one clubs having 05 players from the top50

# there is one players who had 06 contract or has played in 06 clubs


# In[110]:


#projection
G1 = bipartite.projected_graph(G, players)
Gm = bipartite.projected_graph(G, players, 'MultiGraph')
Gw = bipartite.weighted_projected_graph(G, players)

#Let's see the popularity
pop_distribution = Counter(sorted(dict(G1.degree()).values()))
strenght_distribution = Counter(sorted(dict(Gm.degree()).values()))


# In[120]:


#Let's plot the projected degree distribution
def plot_projected_distr(pop_distribution, strenght_distribution):
    
    fig, ax = plt.subplots()
    ax.plot(list(pop_distribution.keys()),list(pop_distribution.values()), 'ro', linestyle = '-', label='Popularity')
    ax.plot(list(strenght_distribution.keys()),list(strenght_distribution.values()), 'yo', linestyle = '-.', label = 'Strength')
    ax.set_ylabel('Frequency', fontsize=16)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    ax.legend(loc='center right', fontsize=12, ncol = 1, frameon = True)
    plt.tight_layout()
    plt.show()
        
plot_projected_distr(pop_distribution, strenght_distribution)
# about even ...players


# In[139]:


# Let's see some numbers
def get_highest(degree_dict, degree_type):
    #order dict by value
    p = Counter(dict(degree_dict))
    print('Highest '  +  degree_type)
    for u, v in p.most_common(5):
        print('%s: %i' % (u,v))
                
get_highest(clubs_pop, ' Club with Number of Contracts/Players')


# In[141]:


get_highest(players_contract, "players with Number of contract")


# In[142]:


get_highest(G.degree(), 'Popularity')


# In[143]:


get_highest(Gm.degree(), "Strength")
# wonder if because of the projection


# In[168]:


#!pip install decorator==5.0.9 then reboot
get_ipython().system('pip install --user decorator==4.3.0 ')
# (ignore warning for availability of newer version)

get_ipython().system('pip install --user networkx==2.3')


# In[173]:


# Visualizing the network...I think those club with zero are causing issue
remove = [clubs for clubs in clubs if G.degree(clubs)==0]
G.remove_nodes_from(remove)

fig1 = plt.figure(1, figsize=(10,10), dpi=200)
pos = nx.spring_layout(G)
colors = {0:'r', 1:'b'}
#nodes
nx.draw_networkx_nodes(G, pos=pos, node_size=[G.degree[node] for node in G], alpha = 0.8, node_color=[colors[G.nodes[node]['bipartite']]for node in G])
#edges
nx.draw_networkx_edges(G, pos=pos, alpha=0.4, edge_color='gray')
plt.axis('off')
plt.show()


# In[174]:


#G = nx.complete_graph()
#nx.draw(G, with_labels=True, font_weight='bold')
#plt.show()


# In[177]:


# centrality
centrality = nx.betweenness_centrality(Gw, normalized=True)
centrality = Counter(nx.betweenness_centrality(Gw, normalized = True, weight='weight'))
print('Highest betweenness centrality')
for u, v in centrality.most_common(5):
    print('%s: %i' % (u,v))


# In[ ]:





#Load the data
df <- read_csv("data/column_3C_weka.csv")
#Just focus on Normal vs Abnormal as the dependent variable
df <- 
  df %>% 
  mutate(class=if_else(class=="Normal","Normal","Abnormal")) %>% 
  filter(degree_spondylolisthesis<=360) #remove crazy outlier
# Convert the features of the data
df.data <- as.matrix(df[1:6])

# Check column means and standard deviations
colMeans(df.data)
summary(df.data)
apply(df.data,2,sd)

# Scale the data: df.scaled
df.scaled <- scale(df.data)
df.scaled
# Calculate the (Euclidean) distances: df.dist
df.dist <- dist(df.scaled)
# Create a hierarchical clustering model: df.hclust
df.hclust <- hclust(d=df.dist,method="complete")
plot(df.hclust)

# Cut tree so that it has 4 clusters: df.hclust.clusters
df.hclust.clusters <- cutree(df.hclust,k=4)

df_clust <- data.frame(df.hclust.clusters)
df_clust$df.hclust.clusters <- as.character(df_clust$df.hclust.clusters)
# Compare cluster membership to actual diagnoses
summary <- df %>% ungroup() %>%  bind_cols(df_clust) %>% 
  rename(cluster=df.hclust.clusters)
names(summary)


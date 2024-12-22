library(readxl)
library(reshape2)
library(ggplot2)

mycol <- c("PCA"="#73A5C3","BayesSpace"="#376689","GraphST"="#EBABA7",
           "SEDR"="#CE4A37","conST"="#9AC567","STAGATE"="#5FA24D", 'SiGra'='#DAB760',
           'xSiGra'="#957651","HERGAST"="#82659C","nMethods"="black")

# simulation
n_clusters <- read_excel("data/leiden_resolution.xlsx",sheet = 1,range = "A2:J10")
sf <- round(max(n_clusters)/max(n_clusters$nMethods))
n_clusters$nMethods <- sf*n_clusters$nMethods
n_clusters$res_aux <- 1:nrow(n_clusters)
n_clusters.l <- melt(n_clusters[,2:ncol(n_clusters)],id.vars = ncol(n_clusters)-1)

ggplot(n_clusters.l,aes(x=res_aux, y=value, colour=variable))+
  geom_line(linewidth=1,alpha=.9) +
  geom_point(size=2,alpha=.8)+
  geom_hline(yintercept = 3,lty='dashed')+
  geom_hline(yintercept = 20,lty='dashed')+
  scale_y_continuous(sec.axis = sec_axis(~ . / sf, name = "Number of viable methods"))+
  scale_x_continuous(breaks = n_clusters$res_aux, labels = n_clusters$resolution)+
  scale_color_manual(values = mycol)+
  labs(x='Resolution',y='Number of clusters',color='Methods')+
  guides(color='none')+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.text = element_text(size=13,colour = 'black'))

ggsave(filename = 'nclusters_simu.svg',width = 6.5, height = 4)

#heatmap
performance <- read_excel("data/leiden_resolution.xlsx",sheet = 1,range = "L2:T10")
performance.l <- melt(performance,id.vars = 1)
performance.l$resolution <- factor(performance.l$resolution)
performance.l$variable <- factor(performance.l$variable,levels=rev(c('GraphST(DIC)','STAGATE(DIC)',"PCA",'conST(DIC)',
                                                                     'SEDR(DIC)','xSiGra(DIC)','SiGra(DIC)','HERGAST')))

ggplot(performance.l, aes(x = resolution, y = variable, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "black") +
  scale_fill_gradient(low = "white", high = "darkred")+
  labs(x='Resolution',y='Methods')+
  guides(fill='none')+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.text.y = element_text(angle = 45,hjust = 1,size=10,colour = 'black'),
        axis.text.x = element_text(size=13,colour = 'black'))

ggsave("heatmap_simu.svg",width = 6.73,height = 4)


# real data
n_clusters <- read_excel("data/leiden_resolution.xlsx",sheet = 2,range = "A2:J12")[3:10,]
sf <- round(max(n_clusters)/max(n_clusters$nMethods))
n_clusters$nMethods <- sf*n_clusters$nMethods
n_clusters$res_aux <- 1:nrow(n_clusters)
n_clusters.l <- melt(n_clusters[,2:ncol(n_clusters)],id.vars = ncol(n_clusters)-1)

ggplot(n_clusters.l,aes(x=res_aux, y=value, colour=variable))+
  geom_line(linewidth=1,alpha=.9) +
  geom_point(size=2,alpha=.8)+
  geom_hline(yintercept = 3,lty='dashed')+
  geom_hline(yintercept = 20,lty='dashed')+
  scale_y_continuous(sec.axis = sec_axis(~ . / sf, name = "Number of viable methods"))+
  scale_x_continuous(breaks = n_clusters$res_aux, labels = n_clusters$resolution)+
  scale_color_manual(values = mycol)+
  labs(x='Resolution',y='Number of clusters',color='Methods')+
  guides(color='none')+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.text = element_text(size=13,colour = 'black'))

ggsave(filename = 'nclusters_real.svg',width = 6.5, height = 4)

#heatmap
performance <- read_excel("data/leiden_resolution.xlsx",sheet = 2,range = "L2:T12")[3:10,]
performance.l <- melt(performance,id.vars = 1)
performance.l$resolution <- factor(performance.l$resolution)
performance.l$variable <- factor(performance.l$variable,levels=rev(c('GraphST(DIC)','STAGATE(DIC)',"PCA",'conST(DIC)',
                                                                     'SEDR(DIC)','xSiGra(DIC)','SiGra(DIC)','HERGAST')))

ggplot(performance.l, aes(x = resolution, y = variable, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "black") +
  scale_fill_gradient(low = "white", high = "darkred")+
  labs(x='Resolution',y='Methods')+
  guides(fill='none')+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.text.y = element_text(angle = 45,hjust = 1,size=10,colour = 'black'),
        axis.text.x = element_text(size=13,colour = 'black'))

ggsave("heatmap_real.svg",width = 6.64,height = 4)

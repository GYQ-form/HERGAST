library(reshape2)
library(ggplot2)
library(readxl)

mycol <- c("PCA"="#73A5C3","BayesSpace"="#376689","GraphST"="#EBABA7",
           "SEDR"="#CE4A37","conST"="#9AC567","STAGATE"="#5FA24D", 'SiGra'='#DAB760',
           'xSiGra'="#957651","HERGAST"="#82659C")

res <- read_excel("data/SMI_Lung.xlsx",range = "A1:E10",sheet = 1)
res.l <- melt(res)

res.l$Methods <- factor(res.l$Methods,levels=c('conST','STAGATE',"BayesSpace",'SEDR'
                                               ,"PCA",'GraphST','xSiGra','SiGra','HERGAST'))

ggplot(res.l,aes(fill=Methods, y=value,x=variable)) +
  geom_bar(stat = "identity",position = 'dodge',
           alpha = 0.8,width = 0.8, col='white') + 
  scale_fill_manual(values = mycol)+
  labs(x='Metrics')+
  guides(fill='none')+
  theme_classic()+
  theme(axis.title = element_blank(),
        axis.text = element_text(size=15,colour = 'black'))

ggsave(filename = 'SMI_metric.svg',width = 7.5, height = 3.5)

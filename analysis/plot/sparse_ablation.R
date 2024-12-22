library(readxl)
library(reshape2)
library(ggplot2)

# mycol <- c("Seurat"="#B6D0E0","SpaGCN"="#4883B1","BayesSpace"="#EBABA7", 
#            "SEDR"="#CE4A37","STAGATE"="#BFDA9F","conST"="#5FA24D", 'STMGCN'='#F1E4C3', 'GraphST'="#B19470",
#            "RGAST"="#82659C")

mycol <- c("dense"="#82659C","sparse"="#C3B5CF")

res1 <- read_excel("data/sparse_ablation.xlsx",range = "B2:C12")
res1.m = apply(res1,2,mean)
res1.sd = apply(res1,2,sd)
res2 <- read_excel("data/sparse_ablation.xlsx",range = "G2:H12")
res2.m = apply(res2,2,mean)
res2.sd = apply(res2,2,sd)
res <- data.frame(mean=c(res1.m,res2.m),sd=c(res1.sd,res2.sd),
                    condition=rep(c('sparse','dense'),2),model=rep(c("Hetero(HERGAST)","Normal"),c(2,2)))

ggplot(res,aes(x=model, y=mean, fill=condition)) +
  geom_bar(stat = "identity",alpha = 0.8,position = position_dodge(),
           width = 0.6,col='white') + 
  geom_errorbar(aes(ymin=mean-sd/sqrt(10), ymax=mean+sd/sqrt(10)),colour="black",
                width=.1,lwd=0.75,position = position_dodge(width = 0.6))+
  scale_fill_manual(values = mycol)+
  labs(x="Graph type",y="Performance (ARI)",fill="Condition")+
  theme_classic()+
  theme(axis.title = element_text(size = 16),
        axis.text = element_text(size=13,colour = 'black'),
        legend.text = element_text(size = 13),
        legend.title = element_text(size = 15))
ggsave(filename = 'sparse_ablation_barplot.svg',width = 5, height = 6.5)

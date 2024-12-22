library(reshape2)
library(ggplot2)
library(readxl)

mycol <- c("PCA"="#73A5C3","BayesSpace"="#376689","GraphST"="#EBABA7",
           "SEDR"="#CE4A37","conST"="#9AC567","STAGATE"="#5FA24D", 'SiGra'='#DAB760',
           'xSiGra'="#957651","HERGAST"="#82659C")


#ARI:1 NMI:2 FMI:3 HS:4
criterion=3

simu1 <- read_excel("data/simulation.xlsx",range = "B3:J13",sheet = criterion)
m <- colMeans(simu1)
res.m <- melt(m)
res.m$n_spots <- 80
res.m$Methods <- row.names(res.m)
Sd <- apply(simu1,2,sd)
res.sd <- melt(Sd)
res.sd$n_spots <- 80
res.sd$Methods <- row.names(res.sd)

simu2 <- read_excel("data/simulation.xlsx",range = "M3:T13",sheet = criterion)
m <- colMeans(simu2)
tmp <- melt(m)
tmp$n_spots <- 160
tmp$Methods <- row.names(tmp)
res.m <- rbind(res.m,tmp)
Sd <- apply(simu2,2,sd)
tmp2 <- melt(Sd)
tmp2$n_spots <- 160
tmp2$Methods <- row.names(tmp2)
res.sd <- rbind(res.sd,tmp2)

simu3 <- read_excel("data/simulation.xlsx",range = "W3:Z13",sheet = criterion)
m <- colMeans(simu3)
tmp <- melt(m)
tmp$n_spots <- 250
tmp$Methods <- row.names(tmp)
res.m <- rbind(res.m,tmp)
Sd <- apply(simu3,2,sd)
tmp2 <- melt(Sd)
tmp2$n_spots <- 250
tmp2$Methods <- row.names(tmp2)
res.sd <- rbind(res.sd,tmp2)

simu4 <- read_excel("data/simulation.xlsx",range = "AC3:AF13",sheet = criterion)
m <- colMeans(simu4)
tmp <- melt(m)
tmp$n_spots <- 360
tmp$Methods <- row.names(tmp)
res.m <- rbind(res.m,tmp)
Sd <- apply(simu4,2,sd)
tmp2 <- melt(Sd)
tmp2$n_spots <- 360
tmp2$Methods <- row.names(tmp2)
res.sd <- rbind(res.sd,tmp2)

simu5 <- read_excel("data/simulation.xlsx",range = "AI3:AK13",sheet = criterion)
m <- colMeans(simu5)
tmp <- melt(m)
tmp$n_spots <- 490
tmp$Methods <- row.names(tmp)
res.m <- rbind(res.m,tmp)
Sd <- apply(simu5,2,sd)
tmp2 <- melt(Sd)
tmp2$n_spots <- 490
tmp2$Methods <- row.names(tmp2)
res.sd <- rbind(res.sd,tmp2)

simu6 <- read_excel("data/simulation.xlsx",range = "AN3:AP13",sheet = criterion)
m <- colMeans(simu6)
tmp <- melt(m)
tmp$n_spots <- 640
tmp$Methods <- row.names(tmp)
res.m <- rbind(res.m,tmp)
Sd <- apply(simu6,2,sd)
tmp2 <- melt(Sd)
tmp2$n_spots <- 640
tmp2$Methods <- row.names(tmp2)
res.sd <- rbind(res.sd,tmp2)

colnames(res.m)[1] <- 'mean'
colnames(res.sd)[1] <- 'sd'
res <- merge(res.m,res.sd)

res$n_spots <- factor(res$n_spots,
                           levels=c('80','160','250','360','490','640'))
res$n_spots <- as.numeric(res$n_spots)
res$Methods <- factor(res$Methods,levels=c("PCA","BayesSpace",'GraphST','SEDR','conST',
                                          'STAGATE','SiGra','xSiGra','HERGAST'))

ggplot(res,aes(x=n_spots, y=mean, colour=Methods))+
  geom_line(linewidth=1,alpha=.9) +
  geom_errorbar(aes(ymin=mean-sd/sqrt(10), ymax=mean+sd/sqrt(10)),colour="black",
                width=.05,lwd=.75,alpha=.6)+
  geom_point(size=2,alpha=.8)+
  labs(col='Methods')+
  scale_color_manual(values = mycol)+
  scale_x_continuous(breaks = c(1:6),
                     labels = c('80','160','250','360','490','640'))+
  theme_classic()+
  theme(axis.title = element_blank(),
        axis.text = element_text(size=13,colour = 'black'),
        legend.text = element_text(size = 13),
        legend.title = element_text(size = 15))

ggsave(filename = 'FMI.svg',width = 8, height = 3.5)


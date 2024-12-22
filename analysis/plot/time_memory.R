library(reshape2)
library(ggplot2)
library(readxl)
library(ggbreak)

mycol <- c("SpatialPCA"="#73A5C3","BayesSpace"="#376689","GraphST"="#EBABA7",
           "SEDR"="#CE4A37","conST"="#9AC567","STAGATE"="#5FA24D", 'SiGra'='#DAB760',
           'xSiGra'="#957651","HERGAST"="#82659C")

################# GPU memory before and after DIC

mem1 <- read_excel("data/memory_time.xlsx",range = "A3:K10",sheet = 1)
colnames(mem1) <- c('Method','5','10','20','40','80','160','250','360','490','640')

mem2 <- read_excel("data/memory_time.xlsx",range = "B19:G26",sheet = 3)
colnames(mem2) <- c('80','160','250','360','490','640')
mem2$Method <- mem1$Method
mem2 <- mem2[-7,]

mem1$DIC <- 'No'
mem2$DIC <- 'Yes'

mem1.l <- melt(mem1)
mem2.l <- melt(mem2)
mem <- rbind(mem1.l,mem2.l)
colnames(mem)[3:4] <- c("n_spots","mem")
mem$n_spots <- factor(mem$n_spots,levels=c('5','10','20','40','80','160','250','360','490','640'))
mem$last <- 0
mem$last[mem$Method=='SEDR' & mem$n_spots=='80' & mem$DIC=='No'] <- 1
mem$last[mem$Method=='conST' & mem$n_spots=='80' & mem$DIC=='No'] <- 1
mem$last[mem$Method=='GraphST' & mem$n_spots=='160' & mem$DIC=='No'] <- 1
mem$last[mem$Method=='SiGra' & mem$n_spots=='10' & mem$DIC=='No'] <- 1
mem$last[mem$Method=='xSiGra' & mem$n_spots=='10' & mem$DIC=='No'] <- 1
mem$last[mem$Method=='STAGATE' & mem$n_spots=='10' & mem$DIC=='No'] <- 1
mem$last[mem$Method=='SEDR' & mem$n_spots=='250' & mem$DIC=='Yes'] <- 1
mem$last[mem$Method=='conST' & mem$n_spots=='250' & mem$DIC=='Yes'] <- 1
mem$last[mem$Method=='GraphST' & mem$n_spots=='160' & mem$DIC=='Yes'] <- 1
mem$last[mem$Method=='SiGra' & mem$n_spots=='250' & mem$DIC=='Yes'] <- 1
mem$last[mem$Method=='xSiGra' & mem$n_spots=='250' & mem$DIC=='Yes'] <- 1
mem$last <- as.character(mem$last)

mem$n_spots <- as.numeric(mem$n_spots)
mem$Method <- factor(mem$Method,levels=c('GraphST','SEDR','conST','STAGATE','SiGra','xSiGra','HERGAST'))
mem[which(mem$Method=='HERGAST'),'DIC'] <- 'Yes'
mem$DIC <- factor(mem$DIC,levels=c('Yes','No'))

ggplot(mem,aes(x=n_spots, y=mem, colour=Method))+
  geom_line(linewidth=1,aes(lty=DIC),alpha=0.9) +
  geom_point(alpha=0.8,stroke=2,mapping = aes(shape = last))+
  geom_hline(yintercept = 80,linetype = "dashed")+
  labs(col='Methods')+
  scale_color_manual(values = mycol)+
  scale_shape_manual(values=c(16,4))+
  scale_x_continuous(breaks = c(1:10),labels = c('5','10','20','40','80','160','250','360','490','640'))+
  theme_classic()+
  guides(shape='none')+
  theme(axis.title = element_blank(),
        axis.text = element_text(size=15,colour = 'black'),
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 16))

ggsave(filename = 'GPU_mem.svg',width = 10, height = 5)

################# CPU memory after DIC & statistical methods

mem3 <- read_excel("data/memory_time.xlsx",range = "B3:G10",sheet = 3)
colnames(mem3) <- c('80','160','250','360','490','640')
mem3$Method = mem1$Method
mem3 <- mem3[-7,]
mem3$DIC <- 'Yes'
mem3.l <- melt(mem3)


mem4 <- read_excel("data/memory_time.xlsx",range = "A3:K6",sheet = 2)
colnames(mem4) <- c('Method','5','10','20','40','80','160','250','360','490','640')
mem4$DIC <- 'No'
mem4.l <- melt(mem4)

mem <- rbind(mem3.l,mem4.l)
colnames(mem)[3:4] <- c("n_spots","mem")
mem[which(mem$Method=='HERGAST'),'DIC'] <- 'Yes'
mem$n_spots <- factor(mem$n_spots,
                            levels=c('5','10','20','40','80','160','250','360','490','640'))
mem$last <- 0
mem$last[mem$Method=='SEDR' & mem$n_spots=='250'] <- 1
mem$last[mem$Method=='conST' & mem$n_spots=='250'] <- 1
mem$last[mem$Method=='GraphST' & mem$n_spots=='160'] <- 1
mem$last[mem$Method=='SiGra' & mem$n_spots=='250'] <- 1
mem$last[mem$Method=='xSiGra' & mem$n_spots=='250'] <- 1
mem$last[mem$Method=='SpatialPCA' & mem$n_spots=='80'] <- 1
mem$last[mem$Method=='BayesSpace' & mem$n_spots=='490'] <- 1
mem$last <- as.character(mem$last)

mem$n_spots <- as.numeric(mem$n_spots)
mem$Method <- factor(mem$Method,levels=c("SpatialPCA","BayesSpace",'GraphST','SEDR','conST',
                                         'STAGATE','SiGra','xSiGra','HERGAST'))
mem$DIC <- factor(mem$DIC,levels=c('Yes','No'))

ggplot(mem,aes(x=n_spots, y=mem, colour=Method))+
  geom_line(linewidth=1,aes(lty=DIC),alpha=0.9) +
  geom_point(stroke=2,alpha=0.8,mapping = aes(shape = last))+
  geom_hline(yintercept = 512,linetype = "dashed")+
  labs(col='Methods')+
  scale_color_manual(values = mycol)+
  scale_shape_manual(values=c(16,4))+
  scale_x_continuous(breaks = c(1:10),labels = c('5','10','20','40','80','160','250','360','490','640'))+
  theme_classic()+
  guides(shape='none')+
  theme(axis.title = element_blank(),
        axis.text = element_text(size=15,colour = 'black'),
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 16))

ggsave(filename = 'CPU_mem.svg',width = 10, height = 5)


################# time consumption after DIC & statistical methods

time1 <- read_excel("data/memory_time.xlsx",range = "K3:P10",sheet = 3)
colnames(time1) <- c('80','160','250','360','490','640')
time1$Method = mem1$Method
time1 <- time1[-7,]
time1$DIC <- 'Yes'
time1.l <- melt(time1)


time2 <- read_excel("data/memory_time.xlsx",range = "N3:X6",sheet = 2)
colnames(time2) <- c('Method','5','10','20','40','80','160','250','360','490','640')
time2$DIC <- 'No'
time2.l <- melt(time2)

time_all <- rbind(time1.l,time2.l)
colnames(time_all)[3:4] <- c("n_spots","time")
time_all[which(time_all$Method=='HERGAST'),'DIC'] <- 'Yes'
time_all$n_spots <- factor(time_all$n_spots,
                      levels=c('5','10','20','40','80','160','250','360','490','640'))
time_all$last <- 0
time_all$last[time_all$Method=='SEDR' & time_all$n_spots=='250'] <- 1
time_all$last[time_all$Method=='conST' & time_all$n_spots=='250'] <- 1
time_all$last[time_all$Method=='GraphST' & time_all$n_spots=='160'] <- 1
time_all$last[time_all$Method=='SiGra' & time_all$n_spots=='250'] <- 1
time_all$last[time_all$Method=='xSiGra' & time_all$n_spots=='250'] <- 1
time_all$last[time_all$Method=='SpatialPCA' & time_all$n_spots=='80'] <- 1
time_all$last[time_all$Method=='BayesSpace' & time_all$n_spots=='490'] <- 1
time_all$last <- as.character(time_all$last)

time_all$n_spots <- as.numeric(time_all$n_spots)
time_all$Method <- factor(time_all$Method,levels=c("SpatialPCA","BayesSpace",'GraphST','SEDR','conST',
                                         'STAGATE','SiGra','xSiGra','HERGAST'))
time_all$DIC <- factor(time_all$DIC,levels=c('Yes','No'))

# time_all$group <- ifelse(time_all$time > 300, "Outlier", "Normal")
# time_all$group[is.na(time_all$group)] <- "Normal"
# time_all$group <- factor(time_all$group,levels=c("Outlier", "Normal"))

ggplot(time_all,aes(x=n_spots, y=log10(time), colour=Method))+
  geom_line(linewidth=1,aes(lty=DIC),alpha=0.9) +
  geom_point(stroke=2,alpha=0.8,mapping = aes(shape = last))+
  geom_hline(yintercept = log10(60),linetype = "dashed")+
  labs(col='Methods')+
  scale_color_manual(values = mycol)+
  scale_shape_manual(values=c(16,4))+
  scale_x_continuous(breaks = c(1:10),labels = c('5','10','20','40','80','160','250','360','490','640'))+
  theme_classic()+
  guides(shape='none')+
  theme(axis.title = element_blank(),
        axis.text = element_text(size=15,colour = 'black'),
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 16),
        strip.text = element_blank())

ggsave(filename = 'time_log.svg',width = 10, height = 5)

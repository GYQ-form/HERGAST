library(readxl)
library(reshape2)
library(ggplot2)

# real data

PC_res <- read_excel("data/PCA_dim.xlsx",sheet=1)

ggplot(PC_res,aes(x=PCs, y=ARI))+
  geom_line(linewidth=1) +
  geom_point(size=2)+
  labs(x='Number of PCs',y='Performance (ARI)')+
  scale_y_continuous(limits = c(0,NA))+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.text = element_text(size=13,colour = 'black'),
        strip.text = element_text(size = 14))
ggsave(filename = 'PCs_res.svg',width = 6, height = 4)


PC_var <- read_excel("data/PCA_dim.xlsx",sheet=5)

ggplot(PC_var,aes(x=PCs, y=var))+
  geom_line(linewidth=1) +
  labs(x='Number of PCs',y='Variance Explained')+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.text = element_text(size=13,colour = 'black'),
        strip.text = element_text(size = 14))
ggsave(filename = 'PCs_var.svg',width = 6, height = 4)


# simulation data

PC_res1 <- read_excel("data/PCA_dim.xlsx",sheet=2)
PC_res1$scale <- '80K'
PC_res2 <- read_excel("data/PCA_dim.xlsx",sheet=3)
PC_res2$scale <- '360K'
PC_res3 <- read_excel("data/PCA_dim.xlsx",sheet=4)
PC_res3$scale <- '640K'
PC_res <- rbind(PC_res1,PC_res2)
PC_res <- rbind(PC_res,PC_res3)
PC_res$scale <- factor(PC_res$scale,levels = c('80K','360K','640K'))

ggplot(PC_res,aes(x=PCs, y=ARI,colour = scale))+
  geom_line(linewidth=1,alpha=0.8) +
  geom_point(size=2,alpha=0.8)+
  labs(x='Number of PCs',y='Performance (ARI)')+
  scale_y_continuous(limits = c(0,NA))+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.text = element_text(size=13,colour = 'black'),
        strip.text = element_text(size = 14),
        legend.title = element_text(size=15,colour = 'black'),
        legend.text = element_text(size=13,colour = 'black'),
        legend.position = c(0.9,0.2))
ggsave(filename = 'PCs_res_simu.svg',width = 6, height = 4)


PC_var1 <- read_excel("data/PCA_dim.xlsx",sheet=6)
PC_var1$scale <- '80K'
PC_var2 <- read_excel("data/PCA_dim.xlsx",sheet=7)
PC_var2$scale <- '360K'
PC_var3 <- read_excel("data/PCA_dim.xlsx",sheet=8)
PC_var3$scale <- '640K'
PC_var <- rbind(PC_var1,PC_var2)
PC_var <- rbind(PC_var,PC_var3)
PC_var$scale <- factor(PC_var$scale,levels = c('80K','360K','640K'))

ggplot(PC_var,aes(x=PCs, y=var,colour = scale))+
  geom_line(linewidth=1,alpha=0.8) +
  labs(x='Number of PCs',y='Variance Explained')+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.text = element_text(size=13,colour = 'black'),
        strip.text = element_text(size = 14),
        legend.title = element_text(size=15,colour = 'black'),
        legend.text = element_text(size=13,colour = 'black'),
        legend.position = c(0.9,0.8))
ggsave(filename = 'PCs_var_simu.svg',width = 6, height = 4)

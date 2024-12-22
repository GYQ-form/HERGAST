library(readxl)
library(reshape2)
library(ggplot2)

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


PC_var <- read_excel("data/PCA_dim.xlsx",sheet=2)

ggplot(PC_var,aes(x=PCs, y=var))+
  geom_line(linewidth=1) +
  labs(x='Number of PCs',y='Variance Explained')+
  # scale_y_continuous(limits = c(0,NA))+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.text = element_text(size=13,colour = 'black'),
        strip.text = element_text(size = 14))
ggsave(filename = 'PCs_var.svg',width = 6, height = 4)


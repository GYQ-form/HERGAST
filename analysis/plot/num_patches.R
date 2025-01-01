library(readxl)
library(reshape2)
library(ggplot2)

resdf1 <- read_excel("data/num_patches.xlsx",sheet = 1)[,-1]
res1 <- melt(resdf1,id.vars = 'Number of spots per patch')
colnames(res1) <- c("Number of spots per patch", "Metric","Measurement")
res1$scale <- '80k'

resdf2 <- read_excel("data/num_patches.xlsx",sheet = 2)[,-1]
res2 <- melt(resdf2,id.vars = 'Number of spots per patch')
colnames(res2) <- c("Number of spots per patch", "Metric","Measurement")
res2$scale <- '360k'

resdf3 <- read_excel("data/num_patches.xlsx",sheet = 3)[,-1]
res3 <- melt(resdf3,id.vars = 'Number of spots per patch')
colnames(res3) <- c("Number of spots per patch", "Metric","Measurement")
res3$scale <- '640k'

res <- rbind(res1,res2)
res <- rbind(res,res3)
res$scale <- factor(res$scale,levels = c('80k','360k','640k'))

# in paper
ggplot(res,aes(x=`Number of spots per patch (log10)`, y=Measurement, colour=Metric))+
  geom_line(linewidth=1) +
  geom_point(size=2)+
  scale_x_continuous(limits = c(NA,4.5))+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.title.y = element_blank(),
        axis.text = element_text(size=13,colour = 'black'),
        strip.text = element_text(size = 14))+
  guides(color = 'none') +
  facet_wrap(~ Metric, scales = "free_y", ncol = 1)
ggsave(filename = 'num_patches.svg',width = 3.8, height = 9)

# in response
ggplot(res,aes(x=`Number of spots per patch`, y=Measurement, colour=scale))+
  geom_line(linewidth=1) +
  geom_point(size=2)+
  scale_x_continuous(breaks = c(0,5000,10000,15000,20000,25000))+
  theme_classic()+
  theme(axis.title = element_text(size=15,colour = 'black'),
        axis.title.y = element_blank(),
        axis.text = element_text(size=13,colour = 'black'),
        strip.text = element_text(size = 14),
        legend.title = element_text(size=15,colour = 'black'),
        legend.text = element_text(size=13,colour = 'black'))+
  facet_wrap(~ Metric, scales = "free_y")
ggsave(filename = 'num_patches_response.svg',width = 11, height = 7)


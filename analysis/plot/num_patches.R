library(readxl)
library(reshape2)
library(ggplot2)

resdf <- read_excel("data/num_patches.xlsx")[,-1]

res <- melt(resdf,id.vars = 'Number of spots per patch')
res$`Number of spots per patch` <- log10(res$`Number of spots per patch`)
colnames(res) <- c("Number of spots per patch (log10)", "Metric","Measurement")

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
  facet_wrap(~ Metric, scales = "free_y")
ggsave(filename = 'num_patches_response.svg',width = 9, height = 6)


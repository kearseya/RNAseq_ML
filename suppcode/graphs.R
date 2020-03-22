library(tidyverse)
library(reshape2)

#Read in data and add column names
filename <- "/home/alex/Desktop/ML/acc_var.csv"
data <- read_csv(filename, col_names = F)
colnames(data) <- c("BaseAcc", "BaseVar", "FeatAcc", "FeatVar", "TuneAcc", "TuneVar", "NumFeat", "NumTrees", "TimeTaken")
#data
summary(data)

#Create dataset for boxplotting of accuracies
accs <- data %>%
  select(c("BaseAcc", "FeatAcc", "TuneAcc"))
colnames(accs) <- c("Baseline", "Feature selection", "Hyperparameter tuning")
boxdata <- melt(accs)

ggplot(boxdata, aes(x=variable, y=value)) +
  geom_boxplot(fill=c("red", "yellow", "green"), color=c("darkred", "orange", "darkgreen")) +
  theme_minimal() +
  labs(x="Test stage", y="Cross validation accuracy")

#Test for normality
#shapiro.test(data$BaseAcc)
shapiro.test(data$FeatAcc)
shapiro.test(data$TuneAcc)

t.test(data$FeatAcc, data$TuneAcc)


#Time vs Size plot
ggplot(data, aes(data$'NumFeat', data$'TimeTaken')) +
  geom_point()
cor(data$NumFeat, data$TimeTaken)


#Read routes data
routesfilename <- "/home/alex/Desktop/ML/routes.csv"
routes <- read.table(routesfilename, header=FALSE, sep=",",
                   col.names=c('Meth1', 'Thresh1', 'Meth2', 'Thresh2', 'Meth3', 'Thresh3', 'Meth4',  'Thresh4'), fill=TRUE)
routes <- as.tibble(routes)
routes <- mutate_all(routes, list(~na_if(.,"")))
methcols <- c('Meth1', 'Meth2', 'Meth3', 'Meth4' )
routes %>% separate_rows(routes, sep = ";") %>%
  mutate(routes = str_trim(routes, side = "both")) %>%

#Extract methods and count number of times used
methods <- routes %>%
  select(c('Meth1', 'Meth2', 'Meth3', 'Meth4' ))
colnames(methods) <- c('Round 1', 'Round 2', 'Round 3', 'Round 4')
methgather <- methods %>%
  gather(na.rm = TRUE)
methgathertable <- as.tibble(table(methgather))
methgathertable <- filter(methgathertable, n>0)
methgathertable <- methgathertable %>% 
  separate_rows(value, sep = ",") %>%
  mutate(value = str_trim(value, side = "both"))

#Plot the number of times each method used for each round
ggplot(methgathertable, aes(x=key, y=n, fill=value)) +
  geom_col() + 
  theme_minimal() +
  scale_color_manual(values=c('#8800E5','#DE0072','#D85000','#9BD200',
                              '#00CB1D','#00BEC5','#000EBF'), aesthetics = "fill") +
  labs(x="Feature select round", y="No. times used", fill="Method")


#Find route taken for highest pecentile models
summary(data$FeatAcc)
topq <- which(data$FeatAcc >= 0.82)
length(topq)

topqmethods <- routes[topq,] %>%
  select(c('Meth1', 'Meth2', 'Meth3', 'Meth4' ))
colnames(topqmethods) <- c('Round 1', 'Round 2', 'Round 3', 'Round 4')
topqmethgather <- topqmethods %>%
  gather(na.rm = TRUE)
topqmethgathertable <- as.tibble(table(methgather))
topqmethgathertable <- filter(methgathertable, n>0)

ggplot(topqmethgathertable, aes(x=key, y=n, fill=value)) +
  geom_col() + 
  theme_minimal() +
  scale_color_manual(values=c('#8800E5','#DE0072','#D85000','#9BD200',
                              '#00CB1D','#00BEC5','#000EBF'), 
                     aesthetics = "fill") +
  labs(x="Feature select round", y="No. times used", fill="Method")

unique(topqmethgathertable$value)


plot(data$FeatAcc, data$TuneAcc)
cor(data$FeatAcc, data$TuneAcc)


cor.test(data$NumFeat, data$TuneAcc)
plot(data$NumFeat, data$FeatAcc)

bscores <- c(0.82,
             0.8933333333333333,
             0.9,
             0.86,
             0.86,
             0.82,
             0.82)
bvars <- c(0.03265986323710903,
           0.17587874611030554,
           0.16329931618554516,
           0.25438378704451886,
           0.25438378704451886,
           0.03265986323710903,
           0.33092463055975624)

plot(data$FeatAcc, data$TuneAcc)
summary(data)
mean(bscores)
mean(bvars)

tree <- c(96, 70, 74, 122, 138/2, 142/2)
feat <- c(390, 220, 234, 466, 179, 317)
#plot(feat, tree)
cor.test(feat, tree)

feat/tree

cor.test(data$NumFeat, data$NumTrees)
data$NumFeat
data$NumTrees
filt <- select(data, c(NumFeat, NumTrees))
filt400 <- filter(filt, NumFeat < 400)
cor.test(filt400$NumFeat, filt400$NumTrees)
plot(filt400$NumFeat, filt400$NumTrees)

mean(filt400$NumFeat/filt400$NumTrees)




top3 <- read_csv("/home/alex/Desktop/ML/top3.csv", col_names = FALSE)

top3 <- top3 %>% group_by(X1) %>% summarise(X2 = sum(X2))
top3 <- arrange(top3, by_group=top3$X2)
colnames(top3) <- c("gene", "val")
ftop3 <- top3 %>% filter(val >= 262)
ftop3$val <- as.integer(ftop3$val)
facgene <- (factor(ftop3$gene, levels = ftop3$gene))

ggplot(ftop3, aes(facgene, val/30, fill=gene)) +
  geom_col() +
  labs(x="Gene", y="Average relative importance") +
  coord_flip() +
  scale_fill_discrete(guide=FALSE) +
  theme_minimal()


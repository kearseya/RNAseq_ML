library(tidyverse)
library(reshape2)

#Read in data and add column names
filename <- "/home/alex/Desktop/ML/acc_var.csv"
data <- read_csv(filename, col_names = F)
colnames(data) <- c("BaseAcc", "BaseVar", "FeatAcc", "FeatVar", "TuneAcc", "TuneVar", "NumFeat", "TimeTaken")
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


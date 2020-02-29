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
boxdata <- melt(accs)

ggplot(boxdata, aes(x=variable, y=value)) +
  geom_boxplot(fill=c("red", "yellow", "green"), color=c("darkred", "orange", "darkgreen")) +
  theme_minimal()

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
#Extract methods and count number of times used
methods <- routes %>%
  select(c('Meth1', 'Meth2', 'Meth3', 'Meth4' ))
colnames(methods) <- c('Round 1', 'Round 2', 'Round 3', 'Round 4')
methgather <- methods %>%
  gather(na.rm = TRUE)
methgathertable <- as.tibble(table(methgather))
methgathertable <- filter(methgathertable, n>0)
#Plot the number of times each method used for each round
ggplot(methgathertable, aes(x=key, y=n, fill=value)) +
  geom_col() + 
  theme_minimal() +
  scale_color_manual(values=c('#8D00E5','#E000B4','#DC0030','#D84D00',
    '#D4C700','#63CF00','#00CB15','#00C788','#008FC3','#001CBF'), 
    aesthetics = "fill") +
  labs(x="Feature select round", y="No. times used", fill="Method")

  
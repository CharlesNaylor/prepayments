---
title: "Bigger Dataset distributions"
output: html_notebook
---

I have 10,000 records plus endogenous data saved to disk now. Let's take a look in R since it's faster and easier to plot stuff.
```{r}
library(tidyverse)
setwd("~/src/LondonMirror/Prepayments/")
sample_data = read_csv("data/samples.csv")
sample_data
```



```{r}
colnames(sample_data)[c(2,3)] = c("pool_number", "as_of_date")
sample_data <- sample_data[,-1]
```

```{r}
sample_data %>% gather(variable, value, -pool_number, -as_of_date) %>%
          ggplot(aes(x=value)) + facet_wrap(~variable, scales = "free") +
          geom_histogram(bins=200)
```

`cato` is bimodal.  What's going on with `next_month_cpr` (endo)?


```{r}
sample_data %>% ggplot(aes(x=next_month_cpr)) + geom_histogram(bins=200)
```

...lots of zeroes.

I
# A function to calculate a simple growth rate for your forecast
calculate_growth <- function(present, past) {
  growth <- ((present - past) / past) * 100
  return(growth)
}

# A function to generate a standardized theme for your research plots
my_research_theme <- function() {
  library(ggplot2)
  theme_minimal() +
    theme(text = element_text(family = "sans", size = 12),
          plot.title = element_text(face = "bold"))
}